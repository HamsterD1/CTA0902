### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 21.16
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090



### Result
```
best-20
test_raw        %SER 88.66 | %CER 30.24 [ 39705 / 131298, 4755 ins, 4630 del, 30320 sub ]

best-5
CH8
test_raw        %SER 88.83 | %CER 30.51 [ 40062 / 131298, 4605 ins, 4903 del, 30554 sub ]
  streaming
  test_raw        %SER 97.26 | %CER 59.32 [ 77885 / 131298, 651 ins, 45750 del, 31484 sub ]


CH4[1,3,5,7]
test_raw        %SER 90.00 | %CER 33.44 [ 43901 / 131298, 4701 ins, 5672 del, 33528 sub ]
  streaming
  test_raw        %SER 97.64 | %CER 63.20 [ 82978 / 131298, 495 ins, 51730 del, 30753 sub ]

CH2[1,5]
test_raw        %SER 91.04 | %CER 36.61 [ 48067 / 131298, 5144 ins, 6264 del, 36659 sub ]
  streaming
  test_raw        %SER 97.91 | %CER 65.85 [ 86460 / 131298, 422 ins, 54914 del, 31124 sub ]


```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
