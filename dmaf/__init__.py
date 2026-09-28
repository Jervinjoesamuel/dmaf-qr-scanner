"""
DMAF — Deep Multimodal Authentication Framework (reference implementation)

A compact, runnable reference implementation of the multimodal QR-code
threat-detection pipeline described in:

  "Deep Multimodal Authentication Framework for Detecting Malicious QR
   Codes in Mobile Payment Systems"

Modalities modelled (matches Eq. 1 in the paper, D = {I, T, U, M, B}):
  I - QR code image features
  T - decoded QR text payload features
  U - URL / domain features
  M - transaction metadata features
  B - mobile behavioral features
"""
