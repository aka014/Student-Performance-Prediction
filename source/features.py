class FeatureList:
    """
    Represents a list of features divided into numerical and categorical groups.

    Attributes:
        numerical (list of str): List of names of numerical features.
        categorical (list of str): List of names of categorical features.
        all (list of str): Combined list of all feature names (categorical + numerical).

    Methods:
        numerical: Getter and setter for numerical features.
        categorical: Getter and setter for categorical features.
        all: Getter and setter for combined feature list.

        WARNING: Once setter for all is used, numerical and categorical features will not be updated. This behaviour is
        not a problem for a way this class is used in the project.
    """

    # Since new class attributes will not be added dynamically, slots are there to speed up access time and memory usage
    __slots__ = ['_numerical', '_categorical', '_all']

    # Class constructor with defaults
    def __init__(self, numerical=None, categorical=None):
        self._numerical = numerical or ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                                        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

        self._categorical = categorical or ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                                            'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                                            'higher', 'internet', 'romantic']

        self._all = self._categorical + self._numerical

    @property
    def numerical(self):
        return self._numerical

    @numerical.setter
    def numerical(self, value):
        if not isinstance(value, list):
            raise TypeError("Numerical features must be a list!")
        self._numerical = value
        self._update_all()

    @property
    def categorical(self):
        return self._categorical

    @categorical.setter
    def categorical(self, value):
        if not isinstance(value, list):
            raise TypeError("Categorical features must be a list!")
        self._categorical = value
        self._update_all()

    @property
    def all(self):
        return self._all

    @all.setter
    def all(self, value):
        if not isinstance(value, list):
            raise TypeError("All features must be a list!")
        self._all = value

    # Mostly I will need to get all column names, so it is important to keep _all updated
    def _update_all(self):
        self._all =  self._categorical + self._numerical
