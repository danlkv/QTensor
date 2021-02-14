
class MeasureName:
    """ A Singletone-ish value for measure name
    Only one can exist for single name, for using in dicts
    """
    _instances = {}
    def __init__(self, name):
        self._name=name

    def __new__(cls, name):
        if cls._instances.get(name) is None:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    @property
    def name(self):
        return self._name
    def __str__(self):
        return self._name
    def __repr__(self):
        return "<" + str(self) + ">"
    def __hash__(self):
        return hash(self._name + __file__)


