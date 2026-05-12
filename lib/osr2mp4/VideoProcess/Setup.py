import numpy as np
from PIL import Image


class FrameInfo:
	"""Replacement for recordclass-based FrameInfo."""
	__slots__ = ['cur_time', 'index_hitobj', 'info_index', 'osr_index',
	             'index_fp', 'obj_endtime', 'x_end', 'y_end', 'break_index']

	def __init__(self, cur_time, index_hitobj, info_index, osr_index,
	             index_fp, obj_endtime, x_end, y_end, break_index):
		self.cur_time = cur_time
		self.index_hitobj = index_hitobj
		self.info_index = info_index
		self.osr_index = osr_index
		self.index_fp = index_fp
		self.obj_endtime = obj_endtime
		self.x_end = x_end
		self.y_end = y_end
		self.break_index = break_index


class CursorEvent:
	"""Replacement for recordclass-based CursorEvent."""
	__slots__ = ['event', 'old_x', 'old_y']

	def __init__(self, event, old_x, old_y):
		self.event = event
		self.old_x = old_x
		self.old_y = old_y


def get_buffer(img, settings):
	np_img = np.frombuffer(img, dtype=np.uint8)
	np_img = np_img.reshape((settings.height, settings.width, 4))
	pbuffer = Image.frombuffer("RGBA", (settings.width, settings.height), np_img, 'raw', "RGBA", 0, 1)
	pbuffer.readonly = False
	return np_img, pbuffer
