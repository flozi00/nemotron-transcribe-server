"""The `vllm` entrypoint: port rewriting, and the collision it must refuse.

The console runs `vllm serve <model> ... --host 0.0.0.0 --port 8000` and probes
8000. The shim takes that port and the engine moves to the internal one — if
the two ever coincide, both listeners fight for one socket and the engine dies
with EADDRINUSE after the shim has already bound it. That is a startup failure
nobody would read as a port clash, so it is asserted here.
"""

import importlib.util
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRYPOINT = os.path.join(HERE, "bin", "vllm")


def load_entrypoint():
    spec = importlib.util.spec_from_loader(
        "vllm_entrypoint",
        importlib.machinery.SourceFileLoader("vllm_entrypoint", ENTRYPOINT),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntrypointTest(unittest.TestCase):
    def setUp(self):
        self.module = load_entrypoint()
        self.execv = mock.patch.object(self.module.os, "execv").start()
        self.popen = mock.patch.object(self.module.subprocess, "Popen").start()
        self.addCleanup(mock.patch.stopall)

    def run_entrypoint(self, argv):
        with mock.patch.object(sys, "argv", ["vllm"] + argv):
            return self.module.main()

    def engine_argv(self):
        self.execv.assert_called_once()
        return self.execv.call_args[0][1]

    def shim_env(self):
        self.popen.assert_called_once()
        return self.popen.call_args[1]["env"]

    def test_console_port_moves_the_engine_aside(self):
        # Exactly what the console emits: --port 8000, the serving port.
        self.run_entrypoint(["serve", "/models/x", "--host", "0.0.0.0", "--port", "8000"])
        argv = self.engine_argv()
        self.assertEqual(argv[argv.index("--port") + 1], str(self.module.INTERNAL_PORT))
        self.assertNotEqual(self.module.INTERNAL_PORT, 8000)
        self.assertEqual(self.shim_env()["SHIM_PORT"], "8000")
        self.assertEqual(
            self.shim_env()["SHIM_UPSTREAM"],
            "http://127.0.0.1:%d" % self.module.INTERNAL_PORT,
        )

    def test_equals_form_is_rewritten_too(self):
        self.run_entrypoint(["serve", "/models/x", "--port=8000"])
        self.assertIn("--port=%d" % self.module.INTERNAL_PORT, self.engine_argv())
        self.assertEqual(self.shim_env()["SHIM_PORT"], "8000")

    def test_other_flags_pass_through_untouched(self):
        flags = ["serve", "/models/x", "--dtype", "bfloat16", "--max-model-len", "65536"]
        self.run_entrypoint(flags + ["--port", "8000"])
        argv = self.engine_argv()
        for flag in flags:
            self.assertIn(flag, argv)

    def test_refuses_when_the_serving_port_is_the_internal_one(self):
        code = self.run_entrypoint(["serve", "/models/x", "--port", str(self.module.INTERNAL_PORT)])
        self.assertEqual(code, 2)
        self.execv.assert_not_called()
        self.popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
