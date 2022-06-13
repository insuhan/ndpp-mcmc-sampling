# Scalable MCMC Sampling for Nonsymmetric Determinantal Point Processes

Python Implmentation for [Scalable MCMC Sampling for Nonsymmetric Determinantal Point Processes]() (ICML 2022)

- The code files include sampling (1) nonymmetric $k$-DPPs ($k$-NDPPs) and (2) nonymmetric DPPs
- The code is based on https://github.com/insuhan/nonsymmetric-dpp-sampling [1]

    [1] Han, I., Gartrell, M., Gillenwater, J., Dohmatob, E., and Kar-basi, A. Scalable Sampling for Nonsymmetric Determi-nantal Point Processes. ICLR 2022 (https://arxiv.org/pdf/2201.08417.pdf)

Usages

- For $k$-NDPP sampling with $k$=10, run

    ```python
    $ python demo_kndpp.py --k 10
    ```

- For NDPP sampling, run

    ```python
    $ python demo_ndpp.py
    ```

