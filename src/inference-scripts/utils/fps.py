import time

class FPS:
    def __init__(self):
        # Total time of execution
        self._start = None
        self._end = None

        # Time of stage of self._num_frames_stage frames
        self._start_stage = None
        self._end_stage = None

        # Total number of frames
        self._num_frames = 0

        # Number of frames of stage
        self._num_frames_stage = 10
        # FPS of stage
        self._fps_stage = 0

    def start(self):
        # Start the timer of execution
        # Start the timer of stage
        self._start = time.time()
        self._start_stage = time.time()
        return self

    def stop(self):
        # Stop de timer of execution
        self._end = time.time()

    def update(self):
        # Increment total number of frames
        # If a stage of 10 frames has been completed,
        # calculate the fps of that stage
        self._num_frames += 1
        if self._num_frames % self._num_frames_stage == 0:
            self._end_stage = time.time()
            self._fps_stage = self._num_frames_stage/(self._end_stage-self._start_stage)
            self._start_stage = time.time()

    def fps(self):
        # If stop() has not been called, return fps of last stage
        # If stop() has been called, return total computed fps
        if self._end == None:
            return self._fps_stage
        return self._num_frames / (self._end - self._start)