from __future__ import annotations

import copy
import logging
import types
import inspect
import typing
from collections import OrderedDict
from enum import Enum
from typing import Any, Optional, Type, List, Dict, Tuple, get_args
from dbus_next.aio import MessageBus
import dbus_next.service

from pyobs.comm import Comm
from pyobs.interfaces import Interface
from pyobs.utils.parallel import Future

log = logging.getLogger(__name__)


class DbusComm(Comm):
    """A Comm class using cbus.

    This Comm class uses dbus for communication between modules.
    """

    __module__ = "pyobs.comm.dbus"

    def __init__(
        self,
        name: str,
        domain: str = "org.pyobs.module",
        *args: Any,
        **kwargs: Any,
    ):
        """Create a new dbus Comm module.

        Args:
            name: Name for export.
            domain: Domain for export.
        """
        Comm.__init__(self, *args, **kwargs)

        # variables
        self._name = name
        self._domain = domain
        self._bus: Optional[MessageBus] = None
        self._dbus_classes: Dict[str, dbus_next.service.ServiceInterface] = {}

    async def open(self) -> None:
        """Creates the dbus connection."""

        # create client
        self._bus = await MessageBus().connect()

        # build and publish classes
        self._build_dbus_classes()
        for interface, obj in self._dbus_classes.items():
            self._bus.export("/" + interface.replace(".", "/"), obj)
            await self._bus.request_name(interface)

        # open Comm
        await Comm.open(self)

    async def close(self) -> None:
        """Close connection."""

        # close parent class
        await Comm.close(self)

        # disconnect from dbus
        if self._bus:
            self._bus.disconnect()

    def _annotation_to_dbus(self, annotation: Any) -> Any:
        if hasattr(annotation, "__origin__") and annotation.__origin__ == list:
            # lists
            return "a" + self._annotation_to_dbus(get_args(annotation)[0])
        elif hasattr(annotation, "__origin__") and annotation.__origin__ == tuple:
            # tuples
            return "(" + "".join([self._annotation_to_dbus(a) for a in get_args(annotation)]) + ")"
        elif hasattr(annotation, "__origin__") and annotation.__origin__ == dict:
            # dicts
            return "a{" + "".join([self._annotation_to_dbus(a) for a in get_args(annotation)]) + "}"
        elif (
            hasattr(annotation, "__origin__")
            and annotation.__origin__ is typing.Union
            and type(None) in get_args(annotation)
        ):
            # optional parameter
            return "s"
        elif inspect.isclass(annotation) and issubclass(annotation, Enum):
            return "s"
        else:
            try:
                return {int: "i", float: "d", str: "s", bool: "b", typing.Any: "s"}[annotation]
            except KeyError:
                raise

    def _build_dbus_signature(self, sig: inspect.Signature) -> inspect.Signature:
        # build list of parameters
        new_params: List[inspect.Parameter] = []
        for name, param in sig.parameters.items():
            # ignore kwargs
            if name == "kwargs":
                continue

            # get new annotation
            annotation = param.annotation if name == "self" else self._annotation_to_dbus(param.annotation)

            # build new param
            p = inspect.Parameter(
                name,
                kind=inspect.Parameter.POSITIONAL_ONLY if name == "self" else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            )

            # store it
            new_params.append(p)

        # return type
        return_annotation = None if sig.return_annotation is None else self._annotation_to_dbus(sig.return_annotation)

        # create new signature
        return inspect.Signature(parameters=new_params, return_annotation=return_annotation)

    def _build_dbus_classes(self) -> None:
        # got a module?
        if self._module is not None:
            # main module
            main_klass = types.new_class(self._name, bases=(dbus_next.service.ServiceInterface,))

            # loop all interfaces features
            for i in self._module.interfaces:
                # create class
                klass_name = f"{self._name}_{i.__name__}"
                interface = f"{self._domain}.{self._name}.interfaces.{i.__name__}"
                klass = types.new_class(klass_name, bases=(dbus_next.service.ServiceInterface,))

                # loop all methods:
                for func_name, func in inspect.getmembers(i, predicate=inspect.isfunction):
                    # get signature
                    sig = inspect.signature(func)
                    dbus_sig = self._build_dbus_signature(sig)

                    # set method
                    my_func = types.MethodType(self._dbus_function_wrapper(func_name, dbus_sig), self)
                    setattr(main_klass, func_name, my_func)

                # initialize it
                self._dbus_classes[interface] = klass(interface)

            # initialize main class
            interface = f"{self._domain}.{self._name}"
            self._dbus_classes[interface] = main_klass(interface)

    def _tuple_to_list(self, sth: Any) -> Any:
        if isinstance(sth, tuple) or isinstance(sth, list):
            return [self._tuple_to_list(a) for a in sth]
        return sth

    def _dbus_function_wrapper(self, method: str, sig: inspect.Signature) -> Any:
        """Function wrapper for dbus methods.

        Args:
            method: Name of method to wrap.

        Returns:
            Wrapper.
        """

        async def inner(this: Any, *args: Any, **kwargs: Any) -> Any:
            # get method of module
            func = getattr(self.module, method)

            # call it
            res = await func(*args, **kwargs)

            # return result
            return self._tuple_to_list(res)

        inner.__signature__ = sig
        # TODO: Nicer way to do this?
        inner.__dict__["__DBUS_METHOD"] = dbus_next.service._Method(inner, method, disabled=False)
        # return dbus_next.service.method(name=method)(inner)
        return inner

    @property
    def name(self) -> Optional[str]:
        """Name of this client."""
        return self._name

    @property
    def clients(self) -> List[str]:
        return []

    async def get_interfaces(self, client: str) -> List[Type[Interface]]:
        return []

    async def _supports_interface(self, client: str, interface: Type[Interface]) -> bool:
        return True

    async def execute(self, client: str, method: str, signature: inspect.Signature, *args: Any) -> Future:
        print("execute", client, method)
        return None


__all__ = ["DbusComm"]
