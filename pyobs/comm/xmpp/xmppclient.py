import logging
import threading
import sleekxmpp
import sleekxmpp.xmlstream
from typing import List

from pyobs.comm.xmpp.xep_0009.rpc import XEP_0009
from pyobs.comm.xmpp.xep_0009_timeout import XEP_0009_timeout


log = logging.getLogger(__name__)


class XmppClient(sleekxmpp.ClientXMPP):
    """XMPP client for pyobs."""

    def __init__(self, jid: str, password: str):
        """Create a new XMPP client.

        An XmppClient handles the actual XMPP communication for the XmppComm module.

        Args:
            jid: Connect to the XMPP server with the given JID.
            password: Password to use for connection.
        """

        sleekxmpp.ClientXMPP.__init__(self, jid, password)

        # stuff
        self._connect_event = threading.Event()
        self._pubsub_callbacks = {}
        self._logs_node = 'logs'
        self._auth_event = threading.Event()
        self._auth_success = False
        self._interface_cache = {}
        self._disco_cache = {}

        # auto-accept invitations
        self.auto_authorize = True

        # register plugins
        self.register_plugin('xep_0009', module=XEP_0009)  # RPC
        self.register_plugin('xep_0009_timeout', module=XEP_0009_timeout)  # RPC timeout
        self.register_plugin('xep_0030')  # Service Discovery
        self.register_plugin('xep_0045')  # Multi-User Chat
        self.register_plugin('xep_0060')  # PubSub
        self.register_plugin('xep_0115')  # Entity Capabilities
        self.register_plugin('xep_0163')  # Personal Eventing Protocol
        self.register_plugin('xep_0199')  # XMPP Ping

        # enable keep alive pings
        self['xep_0199'].enable_keepalive(300, 30)

        # handle session_start and message events
        self.add_event_handler("session_start", self.session_start)
        self.add_event_handler('auth_success', lambda ev: self._auth(True))
        self.add_event_handler('failed_auth', lambda ev: self._auth(False))

    def _get_disco_info(self, jid: str):
        """Return disco info for given jid."""

        # get disco info, is it in cache?
        if jid not in self._disco_cache:
            # request features
            try:
                # get info
                info = self['xep_0030'].get_info(jid=self._jid(jid), cached=False)
                if isinstance(info, sleekxmpp.stanza.iq.Iq):
                    info = info['disco_info']

                # store it
                self._disco_cache[jid] = info.get_features()

            except sleekxmpp.exceptions.IqError:
                return None

        # return it
        return self._disco_cache[jid]

    def get_modules(self, jid: str) -> List[str]:
        """Get list of modules for given jid.

        Args:
            jid: JID to return modules for.

        Returns:
            List of modules.
        """

        # get disco info
        info = self._get_disco_info(jid)

        # get all interface definitions
        prefix = 'pyobs:interface:'
        defs = [i[len(prefix):] for i in info if i.startswith(prefix)]

        # extract module names
        modules = []
        for d in defs:
            if ':' in d:
                modules.append(d.split(':')[0])
        return list(sorted(set(modules)))

    def _jid(self, name: str) -> str:
        """Build full JID from username.

        Args:
            name: Username

        Returns:
            Full JID.
        """
        return '%s@%s/%s' % (name, self.boundjid.domain, self.boundjid.resource) if '@' not in name else name

    def get_interfaces(self, jid: str, module: str = None) -> List[str]:
        """Return list of interfaces for the given JID.

        Args:
            jid: JID to get interfaces for.
            module: Name of module.

        Returns:
            List of interface names
        """

        # get disco info
        info = self._get_disco_info(jid)

        # extract pyobs interfaces
        prefix = 'pyobs:interface:' if module is None else 'pyobs:interface:%s:' % module
        self._interface_cache[(jid, module)] = [i[len(prefix):] for i in info if i.startswith(prefix)]

        # return it
        return self._interface_cache[(jid, module)]

    def supports_interface(self, jid: str, interface) -> bool:
        """Checks, whether a client supports a given interface.

        Args:
            jid: JID of client to check.
            interface: Interface to check.

        Returns:
            True, if client supports interface.
        """
        return self['xep_0030'].supports(jid=jid, feature='pyobs:interface:%s' % interface.__name__)

    def wait_connect(self) -> bool:
        """Wait for client to connect.

        Returns:
            Success or not.
        """

        # wait for auth and check
        self._auth_event.wait()
        if not self._auth_success:
            log.error('Invalid credentials for connecting to XMPP server.')
            return False

        # wait for final connect
        self._connect_event.wait()

        # connected
        return True

    def session_start(self, event):
        """Session start event.

        Args:
            event: The event sent at session start.
        """
        log.info('Connected to server.')

        # send presence and get roster
        self.send_presence()
        self.get_roster()

        # send connected event
        self._connect_event.set()

    def _auth(self, success: bool):
        """Called after authentification.

        Args:
            success: Whether or not the authentification was successful.
        """
        # store and fire
        self._auth_success = success
        self._auth_event.set()

    @property
    def online_clients(self) -> list:
        """Returns list of online clients.

        Returns:
            List of online clients.
        """
        return self._online_clients


__all__ = ['XmppClient']
