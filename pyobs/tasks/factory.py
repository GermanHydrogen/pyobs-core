import threading

from pyobs.object import get_object, get_class_from_string

from .task import Task


class TaskFactoryBase:
    """Base class for all task factories."""

    def __init__(self, comm: 'Comm' = None, observer: 'Observer' = None, vfs: 'VirtualFileSystem' = None,
                 closing_event: threading.Event = None, *args, **kwargs):
        """Initialize a new TaskFactoryBase

        Args:
            comm: Comm object to use.
            observer: Observer.
            vfs: Virtual File System
            closing_event: Event to be set when task should terminates.
        """

        # store it
        self.comm = comm
        self.observer = observer
        self.vfs = vfs
        self.closing = closing_event

    def update_tasks(self):
        """Update list of tasks."""
        pass

    def list(self) -> list:
        """List all tasks from this factory.

        Returns:
            List of all tasks.
        """
        raise NotImplementedError

    def get(self, name: str) -> Task:
        """Returns a single task from the factory.

        Args:
            name: Name of task

        Returns:
            The task object.

        Raises:
            ValueError: If task with given name does not exist.
        """
        raise NotImplementedError

    def create_task(self, name, klass, *args, **kwargs) -> Task:
        """Creates a new task.

        Args:
            name: Name of task.
            klass: Class (itself or its name) of new task.
            *args: Parameters for new task.
            **kwargs: Parameters for new task.

        Returns:
            The new task.
        """

        # is class name given?
        if isinstance(klass, str):
            klass = get_class_from_string(klass)

        # create and return task
        return klass(name=name, comm=self.comm, observer=self.observer, vfs=self.vfs, *args, **kwargs)


class TaskFactory(TaskFactoryBase):
    def __init__(self, factories: dict, *args, **kwargs):
        TaskFactoryBase.__init__(self, *args, **kwargs)

        # create child factories
        self._factories = {}
        if factories is not None:
            # loop all factories
            for name, config in factories.items():
                # create factory
                self._factories[name] = get_object(config, comm=self.comm, observer=self.observer, vfs=self.vfs)

    def list(self) -> list:
        """List all tasks from this factory.

        Returns:
            List of all tasks.
        """

        # collect tasks
        tasks = []
        for name, factory in self._factories.items():
            # get prefix for factory
            prefix = '' if name is None else name + ':'
            tasks.extend([prefix + t for t in factory.list()])

        # return list
        return tasks

    def get(self, name: str) -> Task:
        """Returns a single task from the factory.

        Args:
            name: Name of task

        Returns:
            The task object.

        Raises:
            ValueError: If task with given name does not exist.
        """

        # get prefix
        if ':' in name:
            # split prefix and name
            prefix, name = name.split(':')
        else:
            # no prefix
            prefix = None

        # do we have the prefix?
        if prefix not in self._factories:
            raise ValueError('Given task factory "%s" not found.', prefix)

        # get task from factory
        return self._factories[prefix].get(name)


__all__ = ['TaskFactoryBase', 'TaskFactory']
