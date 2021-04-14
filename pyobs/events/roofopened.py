from .event import Event


class RoofOpenedEvent(Event):
    __module__ = 'pyobs.events'

    def __init__(self):
        Event.__init__(self)


__all__ = ['RoofOpenedEvent']
