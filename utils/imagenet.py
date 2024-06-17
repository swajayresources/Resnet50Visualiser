# -*- coding: utf-8 -*-


import json
from collections.abc import Mapping

from importlib_resources import path

from . import resources


class ImageNetIndex(Mapping):
    """Interface to retrieve ImageNet class indexes from class names.

    This class implements a dictionary like object, aiming to provide an
    easy-to-use look-up table for finding a target class index from an ImageNet
    class name.

    Reference:
        - ImageNet class index: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
        - Synsets: http://image-net.org/challenges/LSVRC/2015/browse-synsets

    Note:
        Class names in `imagenet_class_index.json` has been slightly modified
        from the source due to duplicated class names (e.g. crane). This helps
        make the use of this tool simpler.
    """

    def __init__(self):
        self._index = {}

        with path(resources, 'imagenet_class_index.json') as source_path:
            with open(str(source_path), 'r') as source:
                data = json.load(source)

        for index, (_, class_name) in data.items():
            class_name = class_name.lower().replace('_', ' ')
            self._index[class_name] = int(index)

    """__init__ method: Initializes the ImageNetIndex object.
Initializes self._index as an empty dictionary: This will store the mapping of class names to indexes.
Opens and reads imagenet_class_index.json: Loads the JSON data from the specified resource file.
Parses the JSON data: Converts JSON data into a Python dictionary.
Processes the data: Iterates through the data, converting class names to lowercase, replacing underscores with spaces, and storing them in self._index with their corresponding indexes."""

    def __len__(self):
        return len(self._index)
    
    """__len__ method: Returns the number of items in self._index, allowing the use of len() on ImageNetIndex instances."""

    def __iter__(self):
        return iter(self._index)
    
    """__iter__ method: Returns an iterator over the keys in self._index, enabling iteration over ImageNetIndex instances."""

    def __getitem__(self, phrase):
        if type(phrase) != str:
            raise TypeError('Target class needs to be a string.')

        if phrase in self._index:
            return self._index[phrase]

        partial_matches = self._find_partial_matches(phrase)

        if not any(partial_matches):
            return None
        elif len(partial_matches) > 1:
            raise ValueError('Multiple potential matches found: {}' \
                             .format(', '.join(map(str, partial_matches))))

        target_class = partial_matches.pop()

        return self._index[target_class]
    
    """__getitem__ method: Allows access to self._index using square brackets ([]).
Checks if phrase is a string: Raises a TypeError if not.
Looks for an exact match: Returns the index if phrase is found in self._index.
Finds partial matches: Calls _find_partial_matches to find potential matches.
Handles no matches: Returns None if no matches are found.
Handles multiple matches: Raises a ValueError if multiple matches are found.
Returns the index of the target class: If exactly one partial match is found.
python
"""

    def __contains__(self, key):
        return any(key in name for name in self._index)
    
    """__contains__ method: Implements the in operator, checking if key is contained within any of the class names in self._index."""

    def keys(self):
        return self._index.keys()
    
    """keys method: Returns the keys (class names) of self._index."""

    def items(self):
        return self._index.items()
    
    """items method: Returns the items (class name-index pairs) of self._index."""

    def _find_partial_matches(self, phrase):
        words = phrase.lower().split(' ')

        # Find the intersection between search words and class names to
        # prioritise whole word matches
        # e.g. If words = {'dalmatian', 'dog'} then matches 'dalmatian'

        matches = set(words).intersection(set(self.keys()))

        if not any(matches):
            # Find substring matches between search words and class names to
            # accommodate for fuzzy matches to some extend
            # e.g. If words = {'foxhound'} then matches 'english foxhound'

            matches = [key for word in words for key in self.keys() \
                       if word in key]

        return matches
