from .IStatus import IStatus

import enum


class MotionStatus(enum.Enum):
    """
    Enumerator for moving device status:
        - PARKED means that the device needs to be initialized or positioned or
          moved (depending upon the device; some devices don't need a formal
          initialization); presumedly, this is the safe "off" state.
        - INITIALIZING means that the device is transitioning from a PARKED state
          to an active state but is not yet fully operable.
        - unPARKED is either IDLE (operating but in no particular state) or
          POSITIONED (operating in a well-defined state)
        - SLEWING means that the device is moving to some targeted state (e.g.
          to POSITIONED or TRACKING) but has not yet arrived at that state
        - TRACKING means that the device is moving as commanded
    """
    ABORTING = 'aborting'
    ERROR = 'error'
    IDLE = 'idle'
    INITIALIZING = 'initializing'
    PARKED = 'parked'
    POSITIONED = 'positioned'
    SLEWING = 'slewing'
    TRACKING = 'tracking'
    UNKNOWN = 'unknown'


class IMotionDevice(IStatus):
    """
    Basic interface for all devices that move.

    There are no generic motion methods - these have to be defined in daughter
    interfaces.
    """

    def status(self, *args, **kwargs) -> dict:
        """Returns current status of the motion device.

        Returns:
            dict: A dictionary that should contain at least the following fields:

            IMotionDevice
                Status (str):               Current motion status of device.
        """
        raise NotImplementedError


__all__ = ['IMotionDevice']
