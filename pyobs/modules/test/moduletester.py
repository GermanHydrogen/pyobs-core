import asyncio
import logging
from pydoc import locate
from typing import Dict, Any, Type, List

from pyobs.comm import Comm
from pyobs.modules import Module
from pyobs.object import ProxyType

log = logging.getLogger(__name__)


class ModuleTest:
    def __init__(self, module_name: str, interface: str, method: str, method_args: Dict[str, Any], return_value: str, test_name: str = "UNKNOWN_TEST"):
        self._module_name: str = module_name
        self._module_interface: Type[ProxyType] = locate(interface)

        self._module_method: str = method
        self._method_args: Dict[str, Any] = method_args
        self._return_value: str = return_value
        self._test_name: str = test_name

    @property
    def name(self):
        return self._test_name

    async def __call__(self, comm: Comm, *args, **kwargs):
        module = await comm.proxy(self._module_name, self._module_interface)
        method = getattr(module, self._module_method)
        result = await method(**self._method_args)

        try:
            assert str(result) == self._return_value
        except AssertionError as e:
            log.error(f"{str(result)} != {self._return_value}")
            raise e


class ModuleTester(Module):

    def __init__(self, tests: List[Dict[str, str]], *args, **kwargs):
        self._tests: List[ModuleTest] = [ModuleTest(**x) for x in tests]

        super().__init__(*args, **kwargs)

        self.add_background_task(self.testing, False)

    async def exec_test(self, test: ModuleTest):
        try:
            await test.__call__(self.comm)
        except Exception as e:
            log.error(f'Test "{test.name}" failed!')
            raise e

        log.info(f'Test "{test.name} run successfully!')

    async def testing(self) -> None:

        for test in self._tests:
            await self.exec_test(test)

        await self.close()
