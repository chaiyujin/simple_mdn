import os


class BasicModel():
    def __init__(self, config=None):
        # set config
        self._config = config
        if self._config is None:
            self._config = {}
        # set default save path
        self._default_path = os.path.abspath('.')
        if ('path' in self._config):
            self._default_path = os.path.abspath(self._config['path'])
        self._default_save_path = os.path.join(
            self._default_path, 'save'
        )

    def save(self, sess, name='best', step=0):
        if not os.path.exists(self._default_save_path):
            os.makedirs(self._default_save_path)
        path = os.path.join(self._default_save_path, name)
        self._saver.save(sess, path, global_step=step)

    def load(self, sess, name='best', step=0):
        path = os.path.join(self._default_save_path, name + '-' + str(step))
        assert(os.path.exists(path + '.index'))
        self._saver.restore(sess, path)

    @property
    def loss_fn(self):
        raise NotImplementedError()

    # give the placeholder dictionary
    @property
    def placeholder_dict(self):
        raise NotImplementedError()

    def error_rate(self, pred, true):
        raise NotImplementedError()
