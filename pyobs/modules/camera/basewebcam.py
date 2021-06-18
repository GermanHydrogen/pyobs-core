import io
import logging
import threading
import time
import asyncio
import numpy as np
import tornado
import PIL.Image

from pyobs.modules import Module
from pyobs.interfaces import ICameraExposureTime, IWebcam
from .basecam import BaseCam
from ...images import Image

log = logging.getLogger(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Title</title>
  </head>
  <body>
    <img src="/video">
  </body>
</html>

"""

class MainHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def get(self):
        self.write(INDEX_HTML)


class VideoHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def get(self):
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Pragma', 'no-cache')
        self.set_header('Content-Type', 'multipart/x-mixed-replace;boundary=--jpgboundary')
        self.set_header('Connection', 'close')

        self.served_image_timestamp = time.time()
        my_boundary = "--jpgboundary"
        while True:
            try:
                # Generating images for mjpeg stream and wraps them into http resp
                interval = 0.1
                if self.served_image_timestamp + interval < time.time():
                    image = self.application.image_jpeg
                    self.write(my_boundary)
                    self.write("Content-type: image/jpeg\r\n")
                    self.write("Content-length: %s\r\n\r\n" % len(image))
                    self.write(image)
                    self.served_image_timestamp = time.time()
                    yield self.flush()
                else:
                    yield tornado.gen.sleep(interval)
            except tornado.iostream.StreamClosedError:
                # stream closed
                break


class ImageHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def get(self):
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Pragma', 'no-cache')
        self.set_header('Content-Type', 'image/fits')
        image = self.application.image_fits
        self.set_header("Content-length", len(image))
        self.write(image)


class BaseWebcam(BaseCam, tornado.web.Application, IWebcam, ICameraExposureTime):
    """Base class for all webcam modules."""
    __module__ = 'pyobs.modules.camera'

    def __init__(self, http_port: int = 37077, interval: float = 0.5, *args, **kwargs):
        """Creates a new BaseWebcam.

        Args:
            http_port: HTTP port for webserver.
            interval: Min interval for grabbing images.
        """
        BaseCam.__init__(self, *args, **kwargs)

        # store
        self._io_loop = None
        self._lock = threading.RLock()
        self._is_listening = False
        self._port = http_port
        self._interval = interval

        # init tornado web server
        tornado.web.Application.__init__(self, [
            (r"/", MainHandler),
            (r"/video", VideoHandler),
            (r"/image.fits", ImageHandler),
        ])
        self._image_fits = None
        self._image_jpeg = None
        self._image_time = None

        # add thread func
        self.add_thread_func(self._http, False)

    def open(self):
        """Open module."""
        Module.open(self)

    def close(self):
        """Close server."""

        # close io loop and parent
        self._io_loop.add_callback(self._io_loop.stop)
        Module.close(self)

    @property
    def opened(self) -> bool:
        """Whether the server is started."""
        return self._is_listening

    def _http(self):
        """Thread function for the web server."""

        # create io loop
        asyncio.set_event_loop(asyncio.new_event_loop())
        self._io_loop = tornado.ioloop.IOLoop.current()
        self._io_loop.make_current()

        # start listening
        log.info('Starting HTTP server on port %d...', self._port)
        self.listen(self._port, max_buffer_size=self._max_file_size, max_body_size=self._max_file_size)

        # start the io loop
        self._is_listening = True
        self._io_loop.start()

    @property
    def image_fits(self):
        with self._lock:
            return self._image_fits

    @property
    def image_jpeg(self):
        with self._lock:
            return self._image_jpeg

    def _set_image(self, data: np.ndarray):
        """Create FITS and JPEG images from data."""

        # check interval
        now = time.time()
        if self._image_time is not None and now < self._image_time + self._interval:
            return
        self._image_time = now

        # create FITS file
        image_fits = Image.from_bytes(data)

        # write to buffer and return it
        with io.BytesIO() as output:
            PIL.Image.fromarray(data).save(output, format="JPG")
            image_jpeg = output.getvalue()

        # store both
        with self._lock:
            self._image_time = now
            self._image_fits = image_fits
            self._image_jpeg = image_jpeg

    def wait_for_frame(self, *args, **kwargs):
        """Wait for next frame that starts after this method has been called."""
        raise NotImplementedError

    def get_last_frame(self, *args, **kwargs) -> str:
        """Returns filename of last frame.

        Returns:
            Filename for last exposure.
        """
        raise NotImplementedError



__all__ = ['BaseWebcam']
