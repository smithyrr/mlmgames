_index1 = _ctrl tvAdd [[], "Item1"]; // Adds "Item1" to main branch. Item path [_index1] or [0]  
_index1_1 = _ctrl tvAdd [[0], "Item1_1"]; // Adds "Item1_1" to "Item1". Item path [0,_index1_1] or [0,0]
_index1_2 = _ctrl tvAdd [[0], "Item1_2"]; // Adds "Item1_2" to "Item1". Item path [0,_index1_2] or [0,1]
_index2_1 = _ctrl tvAdd [[0,0], "Item2_1"]; // Adds "Item2_1" to "Item1_1". Item path [0,0,_index2_1] or [0,1,0]