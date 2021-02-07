## Changelog

### v0.13 ()
* Added a Telegram bot module.
* Added a module for a Kiosk mode, in which pictures are published on a webpage.
* Added new IImageFormats interface for cameras that support multiple ones (e.g. grayscale and color).


### v0.12 (2021-01-01)

* Changed PyObsModule to Module.
* Removed possibility for network configs.
* Added MultiModule, which allows for multiple modules in one process.
* Flat scheduler: add options for readout times.
* New OnlineReduction module for reduction during the night.
* Fixed bug that sometimes appears in the interface caching for Comm.
* LcoTaskArchive: added MoonSeparationConstraint, fixed AirmassConstraint.
* Optimized Scheduler by only scheduling blocks that actually have a window in the given range.
* Added module Seeing that extracts FWHMs from the catalogs in reduced images and calculated a median seeing.
* Introduced concept of Publishers, which can be used to publish data to log, CSV, and hopefully later, database, 
  web, etc.
* Created new Object class that handles most of what Module did before so that Module only adds module specific stuff.
* Added some convenience methods for reading/writing files to VFS.
* Added new IConfig interface which is implemented in every module and allows remote access to config parameters 
  (if getter/setters are implemented).
* Removed count parameter from ICamera.expose().
* Removed exposure_time parameter from ICamera.expose() and introduced ICameraExposureTime interface.
* Removed image_type parameter from ICamera.expose() and introduced IImageType.
* Moved ImageType enumerator from ICamera to utils.enums.


### v0.11 (2020-10-18)

* Major changes to robotic system based on LCO portal.
* Setting filter/window/binning in acquisition.
* Added WaitForMotion and Follow mixins.
* Added support for flats that don't directly scale with binning.
* New module for acoustic warning when autonomous modules are running.
* Improved SepPhotometry by calculating columns used also by LCO.
* New interface for Lat/Lon telescopes, e.g. solar telescopes.


### v0.10 (2020-05-05)

* Re-factored acquisition modules and added one based on astrometry.
* Added combine_binnings parameter to FlatFielder, which triggers, whether to use one function for all binnings or not
* Added get_current_weather() to IWeather
* New FlatFieldPointing module that can move telescope to a flatfield pointing
* Changed requirements in setup.py and put packages that are only required by a server module into [full]
* Removed HTTP proxy classes
* Some new mixins


### v0.9 (2020-03-06)

* working on robotic system based on LCO portal


### v0.8 (2019-11-17)

* Added module for bright star acquisition.
* Added and changed some FITS header keywords.
* Added module for flat-fielding.
* Changed some interfaces.
* Added basic pipeline.
* Started with code that will be used for a full robotic mode.
* Re-organized auto-guiding modules.
* and many more...