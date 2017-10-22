# Projgrad: A python library for projected gradient optimization

Python provides general purpose optimization routines via its [`scipy.optimize`](http://docs.scipy.org/doc/scipy/reference/optimize.html) package. For specific problems simple first-order methods such as projected gradient optimization might be more efficient, especially for large-scale optimization and low requirements on solution accuracy. This package aims to provide implementations of projected gradient optimization routines. To ease usage these routines follow a call syntax compatible with `scipy.optimize`.

## Support and contributing

For bug reports and enhancement requests use the [Github issue tool](http://github.com/andim/projgrad/issues/new), or (even better!) open a [pull request](http://github.com/andim/projgrad/pulls) with relevant changes. If you have any questions don't hesitate to contact me by email (andimscience@gmail.com) or Twitter ([@andimscience](http://twitter.com/andimscience)).

You can run the testsuite by running `pytest` in the top-level directory.

You are cordially invited to [contribute](https://github.com/andim/projgrad/blob/master/CONTRIBUTING.md) to the further development of projgrad!
