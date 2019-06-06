# pylint: disable=unused-argument, no-self-use


class shotDetector(object):
    """ Base class to inheret from when implementing a shot detection algorithm.

    Also see the implemented shot detectors in the shotdetect.detectors module
    to get an idea of how a particular detector can be created.
    """

    stats_manager = None
    """ Optional :py:class:`StatsManager <shotdetect.stats_manager.StatsManager>` to
    use for caching frame metrics to and from."""

    _metric_keys = []
    """ List of frame metric keys to be registered with the :py:attr:`stats_manager`,
    if available. """

    cli_name = 'detect-none'
    """ Name of detector to use in command-line interface description. """

    def is_processing_required(self, frame_num):
        # type: (int) -> bool
        """ Is Processing Required: Test if all calculations for a given frame are already done.

        Returns:
            bool: False if the shotDetector has assigned _metric_keys, and the
            stats_manager property is set to a valid StatsManager object containing
            the required frame metrics/calculations for the given frame - thus, not
            needing the frame to perform shot detection.

            True otherwise (i.e. the frame_img passed to process_frame is required
            to be passed to process_frame for the given frame_num).
        """
        return not self._metric_keys or not (
            self.stats_manager is not None and
            self.stats_manager.metrics_exist(frame_num, self._metric_keys))


    def get_metrics(self):
        # type: () -> List[str]
        """ Get Metrics:  Get a list of all metric names/keys used by the detector.

        Returns:
            List[str]: A list of strings of frame metric key names that will be used by
            the detector when a StatsManager is passed to process_frame.
        """
        return self._metric_keys


    def process_frame(self, frame_num, frame_img):
        # type: (int, numpy.ndarray) -> Tuple[bool, Union[None, List[int]]
        """ Process Frame: Computes/stores metrics and detects any shot changes.

        Prototype method, no actual detection.

        Returns:
            List[int]: List of frame numbers of cuts to be added to the cutting list.
        """
        return []


    def post_process(self, frame_num):
        # type: (int) -> List[int]
        """ Post Process: Performs any processing after the last frame has been read.

        Prototype method, no actual detection.

        Returns:
            List[int]: List of frame numbers of cuts to be added to the cutting list.
        """
        return []

