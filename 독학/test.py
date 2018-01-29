# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:47:06 2017

@author: jsc5565
"""

mstr = "FJCCQV ZPNYIX RWUATO WHULQO KADY"
key = "SIBAb"
for i in range(5):
    print(chr(ord(mstr[i])+ord(key[i])))