# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ahead-of-time (XAOT) compilation of `MaxTextTrainingEngine`'s training step.

`trainers/pre_train/train_compile.py` does this for `train.py`'s single fused
`train_step`. The engine splits the same work across three kernels -- one forward/backward
for the first micro-batch of an update, an accumulating one for every later micro-batch,
and the optimizer update -- so this compiles all three and reports the cost and memory of
each.

Nothing is materialized: the weights, the optimizer moments and the batch are all
`jax.ShapeDtypeStruct`s, and the device mesh is a topology description rather than
hardware. So a v5e-256 configuration can be compiled from a workstation, and an
out-of-memory configuration reports the same `RESOURCE_EXHAUSTED` it would report on the
target -- before the target is booked.

The kernels come from `MaxTextTrainingEngine.lower()`, so the HLO compiled here is the HLO
training runs, byte for byte; `tests/post_training/unit/maxtext_engine_xaot_test.py`
asserts it.

Example, qwen3-0.6b on four v6e chips:

  python3 -m maxtext.training_engine.maxtext_engine_compile src/maxtext/configs/base.yml \
    model_name=qwen3-0.6b run_name=engine_aot_qwen3 \
    compile_topology=v6e-4 compile_topology_num_slices=1 \
    per_device_batch_size=4 max_target_length=2048 \
    ici_fsdp_parallelism=4 attention=flash enable_checkpointing=false

No tokenizer, no data pipeline and no checkpoint are needed or touched. Add
`compiled_trainstep_file=/tmp/engine_qwen3.pickle` to serialize the executables; each
kernel is written to its own file, suffixed with the kernel name.
"""

import os
from typing import Any, Sequence

from absl import app
import jax
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train_compile as pre_train_compile
from maxtext.training_engine import maxtext_engine
from maxtext.utils import gcs_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils

# In the order they run.
KERNEL_NAMES = ("fwd_bwd", "fwd_bwd_accum", "update")

# `dump_hlo`'s two filters -- the `--xla_dump_hlo_module_re` XLA is given, and the substring
# `upload_dump` keeps -- both default to `jit_train_step`, the name `train.py` gives its fused
# step. The engine's kernels lower as `jit_first_kernel`, `jit_accum_kernel` and
# `jit__update_kernel`, so on the pre-train defaults the dump comes back empty.
HLO_DUMP_DEFAULTS = {
    "dump_hlo_local_module_name": "jit_.*kernel",
    "dump_hlo_module_name": "kernel",
}


def with_engine_hlo_dump_defaults(argv: Sequence[str]) -> list[str]:
  """Returns `argv` with the HLO dump filters pointed at the kernels, if a dump was asked for.

  Applied to `argv` rather than to the config because `pyconfig.initialize` bakes the regex
  into `XLA_FLAGS` on its way out, and only when `dump_hlo` is on because it bakes it in
  either way: widening the regex unconditionally would leave every compile writing an HLO
  dump nobody asked for.
  """
  given = dict(arg.split("=", 1) for arg in argv if "=" in arg)
  if given.get("dump_hlo", "").strip().lower() not in ("true", "1"):
    return list(argv)
  return list(argv) + [f"{key}={value}" for key, value in HLO_DUMP_DEFAULTS.items() if key not in given]


def get_shaped_micro_batch(config: pyconfig.HyperParameters) -> dict[str, jax.ShapeDtypeStruct]:
  """Returns the abstract batch one `fwd_bwd` call is given.

  `maxtext_utils.get_shaped_batch` shapes the *global* batch, because `train.py`'s fused
  step folds gradient accumulation inside itself. The engine does not: its caller drives
  one `fwd_bwd` per micro-batch. Compiling against the global batch would size every
  activation by the accumulation factor and report a peak memory no step ever reaches.

  No `batch_sharding`: a driver feeds host arrays, which arrive uncommitted and are placed
  by the kernel's own `in_shardings`.
  """
  shaped_batch = maxtext_utils.get_shaped_batch(config)
  micro_batch_size = int(config.micro_batch_size_to_train_on)

  def to_micro_batch(aval: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    if not aval.shape or aval.shape[0] == micro_batch_size:
      return aval
    return jax.ShapeDtypeStruct((micro_batch_size,) + aval.shape[1:], aval.dtype)

  return {key: to_micro_batch(aval) for key, aval in shaped_batch.items()}


def compile_engine_kernels(config: pyconfig.HyperParameters, topology_mesh: jax.sharding.Mesh) -> dict[str, Any]:
  """Lowers and compiles every kernel the engine runs, on `topology_mesh`.

  Returns:
    `{kernel name: jax.stages.Compiled}`, keyed by `KERNEL_NAMES`.
  """
  engine = maxtext_engine.MaxTextTrainingEngine(config, mesh=topology_mesh, materialize_weights=False)
  lowered = engine.lower(get_shaped_micro_batch(config))
  compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
  return {name: lowered[name].compile(compiler_options=compiler_options) for name in KERNEL_NAMES}


def kernel_save_path(compiled_trainstep_file: str, kernel_name: str) -> str:
  """Returns where one kernel's serialized executable goes.

  Three executables and one configured path, so the kernel name goes in the stem rather
  than the three overwriting each other: `/tmp/engine.pickle` becomes
  `/tmp/engine_fwd_bwd.pickle`.
  """
  stem, extension = os.path.splitext(compiled_trainstep_file)
  return f"{stem}_{kernel_name}{extension}"


def main(argv: Sequence[str]) -> None:
  """Compiles the engine's kernels for `compile_topology` and reports what they cost."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting training_engine/maxtext_engine_compile.py...", flush=True)

  config = pyconfig.initialize(with_engine_hlo_dump_defaults(argv))
  pre_train_compile.validate_config(config)
  if config.enable_diloco:
    raise NotImplementedError(
        "enable_diloco is not supported by the engine's AOT path; MaxTextTrainingEngine has no DiLoCo "
        "outer step, so the numbers reported here would describe a different computation."
    )

  topology_mesh = pre_train_compile.get_topology_mesh(config)

  # After the topology is built, so this does not initialize the local backend first.
  max_utils.print_system_information()

  print("Jitting and compiling the engine's kernels...", flush=True)
  compiled = compile_engine_kernels(config, topology_mesh)
  print("Jitting and compilation complete!", flush=True)

  for name in KERNEL_NAMES:
    print(f"--- {name} ---")
    print(f"Cost analysis: {compiled[name].cost_analysis()}")
    print(f"Memory analysis: {compiled[name].memory_analysis()}")

  if config.compiled_trainstep_file != "":
    for name in KERNEL_NAMES:
      save_path = kernel_save_path(config.compiled_trainstep_file, name)
      pre_train_compile.save_compiled(compiled[name], save_path)
      print(f"Successfully saved compiled {name} kernel as {save_path}")

  print("Finished training_engine/maxtext_engine_compile.py successfully!", flush=True)

  if config.dump_hlo:
    # `upload_dump` deletes the directory it uploaded, and does not survive being handed one
    # XLA never wrote; say which filter was too narrow instead of raising from the rmtree.
    if not os.path.isdir(config.dump_hlo_local_dir):
      raise FileNotFoundError(
          f"dump_hlo is set but XLA wrote nothing to {config.dump_hlo_local_dir}: "
          f"dump_hlo_local_module_name={config.dump_hlo_local_module_name!r} matched none of the engine's "
          f"kernels (jit_first_kernel, jit_accum_kernel, jit__update_kernel)."
      )
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


if __name__ == "__main__":
  app.run(main)
