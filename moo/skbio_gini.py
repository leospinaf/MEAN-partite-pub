# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import textwrap

#from skbio.diversity._util import _validate_counts_vector
#from skbio.util._decorator import experimental

class _state_decorator:
    """ Base class for decorators of all public functionality.
    """

    _required_kwargs = ()

    def _get_indentation_level(self, docstring_lines,
                               default_existing_docstring=4,
                               default_no_existing_docstring=0):
        """ Determine the level of indentation of the docstring to match it.

            The indented content after the first line of a docstring can
            differ based on the nesting of the functionality being documented.
            For example, a top-level function may have its "Parameters" section
            indented four-spaces, but a method nested under a class may have
            its "Parameters" section indented eight spaces. This function
            determines the indentation level of the first non-whitespace line
            following the initial summary line.
        """
        # if there is no existing docstring, return the corresponding default
        if len(docstring_lines) == 0:
            return default_no_existing_docstring

        # if there is an existing docstring with only a single line, return
        # the corresponding default
        if len(docstring_lines) == 1:
            return default_existing_docstring

        # find the first non-blank line (after the initial summary line) and
        # return the number of leading spaces on that line
        for line in docstring_lines[1:]:
            if len(line.strip()) == 0:
                # ignore blank lines
                continue
            else:
                return len(line) - len(line.lstrip())

        # if there is an existing docstring with only a single non-whitespace
        # line, return the corresponding default
        return default_existing_docstring

    def _update_docstring(self, docstring, state_desc,
                          state_desc_prefix='State: '):
        # Hande the case of no initial docstring
        if docstring is None:
            return "%s%s" % (state_desc_prefix, state_desc)

        docstring_lines = docstring.split('\n')
        docstring_content_indentation = \
            self._get_indentation_level(docstring_lines)

        # wrap lines at 79 characters, accounting for the length of
        # docstring_content_indentation and start_desc_prefix
        len_state_desc_prefix = len(state_desc_prefix)
        wrap_at = 79 - (docstring_content_indentation + len_state_desc_prefix)
        state_desc_lines = textwrap.wrap(state_desc, wrap_at)
        # The first line of the state description should start with
        # state_desc_prefix, while the others should start with spaces to align
        # the text in this section. This is for consistency with numpydoc
        # formatting of deprecation notices, which are done using the note
        # Sphinx directive.
        state_desc_lines[0] = '%s%s%s' % (' ' * docstring_content_indentation,
                                          state_desc_prefix,
                                          state_desc_lines[0])
        header_spaces = ' ' * (docstring_content_indentation +
                               len_state_desc_prefix)
        for i, line in enumerate(state_desc_lines[1:], 1):
            state_desc_lines[i] = '%s%s' % (header_spaces, line)

        new_doc_lines = '\n'.join(state_desc_lines)
        docstring_lines[0] = '%s\n\n%s' % (docstring_lines[0], new_doc_lines)
        return '\n'.join(docstring_lines)

    def _validate_kwargs(self, **kwargs):
        for required_kwarg in self._required_kwargs:
            if required_kwarg not in kwargs:
                raise ValueError('%s decorator requires parameter: %s' %
                                 (self.__class__, required_kwarg))


class stable(_state_decorator):
    """ State decorator indicating stable functionality.

    Used to indicate that public functionality is considered ``stable``,
    meaning that its API will be backward compatible unless it is deprecated.
    Decorating functionality as stable will update its doc string to indicate
    the first version of scikit-bio when the functionality was considered
    stable.

    Parameters
    ----------
    as_of : str
        First release version where functionality is considered to be stable.

    See Also
    --------
    experimental
    deprecated

    Examples
    --------
    >>> @stable(as_of='0.3.0')
    ... def f_stable():
    ...     \"\"\" An example stable function.
    ...     \"\"\"
    ...     pass
    >>> help(f_stable)
    Help on function f_stable in module skbio.util._decorator:
    <BLANKLINE>
    f_stable()
        An example stable function.
    <BLANKLINE>
        State: Stable as of 0.3.0.
    <BLANKLINE>
    """

    _required_kwargs = ('as_of', )

    def __init__(self, *args, **kwargs):
        self._validate_kwargs(**kwargs)
        self.as_of = kwargs['as_of']

    def __call__(self, func):
        state_desc = 'Stable as of %s.' % self.as_of
        func.__doc__ = self._update_docstring(func.__doc__, state_desc)
        return func


class experimental(_state_decorator):
    """ State decorator indicating experimental functionality.

    Used to indicate that public functionality is considered experimental,
    meaning that its API is subject to change or removal with little or
    (rarely) no warning. Decorating functionality as experimental will update
    its doc string to indicate the first version of scikit-bio when the
    functionality was considered experimental.

    Parameters
    ----------
    as_of : str
        First release version where feature is considered to be experimental.

    See Also
    --------
    stable
    deprecated

    Examples
    --------
    >>> @experimental(as_of='0.3.0')
    ... def f_experimental():
    ...     \"\"\" An example experimental function.
    ...     \"\"\"
    ...     pass
    >>> help(f_experimental)
    Help on function f_experimental in module skbio.util._decorator:
    <BLANKLINE>
    f_experimental()
        An example experimental function.
    <BLANKLINE>
        State: Experimental as of 0.3.0.
    <BLANKLINE>

    """

    _required_kwargs = ('as_of', )

    def __init__(self, *args, **kwargs):
        self._validate_kwargs(**kwargs)
        self.as_of = kwargs['as_of']

    def __call__(self, func):
        state_desc = 'Experimental as of %s.' % self.as_of
        func.__doc__ = self._update_docstring(func.__doc__, state_desc)
        return func

def _validate_counts_vector(counts, suppress_cast=False):
    """Validate and convert input to an acceptable counts vector type.

    Note: may not always return a copy of `counts`!

    """
    counts = np.asarray(counts)
    try:
        if not np.all(np.isreal(counts)):
            raise Exception
    except Exception:
        raise ValueError("Counts vector must contain real-valued entries.")
    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    elif (counts < 0).any():
        raise ValueError("Counts vector cannot contain negative values.")

    return counts

@experimental(as_of="0.4.0")
def gini_index(data, method='rectangles'):
    r"""Calculate the Gini index.

    The Gini index is defined as

    .. math::

       G=\frac{A}{A+B}

    where :math:`A` is the area between :math:`y=x` and the Lorenz curve and
    :math:`B` is the area under the Lorenz curve. Simplifies to :math:`1-2B`
    since :math:`A+B=0.5`.

    Parameters
    ----------
    data : 1-D array_like
        Vector of counts, abundances, proportions, etc. All entries must be
        non-negative.
    method : {'rectangles', 'trapezoids'}
        Method for calculating the area under the Lorenz curve. If
        ``'rectangles'``, connects the Lorenz curve points by lines parallel to
        the x axis. This is the correct method (in our opinion) though
        ``'trapezoids'`` might be desirable in some circumstances. If
        ``'trapezoids'``, connects the Lorenz curve points by linear segments
        between them. Basically assumes that the given sampling is accurate and
        that more features of given data would fall on linear gradients between
        the values of this data.

    Returns
    -------
    double
        Gini index.

    Raises
    ------
    ValueError
        If `method` isn't one of the supported methods for calculating the area
        under the curve.

    Notes
    -----
    The Gini index was introduced in [1]_. The formula for
    ``method='rectangles'`` is

    .. math::

       dx\sum_{i=1}^n h_i

    The formula for ``method='trapezoids'`` is

    .. math::

       dx(\frac{h_0+h_n}{2}+\sum_{i=1}^{n-1} h_i)

    References
    ----------
    .. [1] Gini, C. (1912). "Variability and Mutability", C. Cuppini, Bologna,
       156 pages. Reprinted in Memorie di metodologica statistica (Ed. Pizetti
       E, Salvemini, T). Rome: Libreria Eredi Virgilio Veschi (1955).

    """
    # Suppress cast to int because this method supports ints and floats.
    data = _validate_counts_vector(data, suppress_cast=True)
    lorenz_points = _lorenz_curve(data)
    B = _lorenz_curve_integrator(lorenz_points, method)
    return 1 - 2 * B

## taken verbatim from skibio repo

def _lorenz_curve(data):
    """Calculate the Lorenz curve for input data.

    Notes
    -----
    Formula available on wikipedia.

    """
    sorted_data = np.sort(data)
    Sn = sorted_data.sum()
    n = sorted_data.shape[0]
    return np.arange(1, n + 1) / n, sorted_data.cumsum() / Sn


def _lorenz_curve_integrator(lc_pts, method):
    """Calculates the area under a Lorenz curve.

    Notes
    -----
    Could be utilized for integrating other simple, non-pathological
    "functions" where width of the trapezoids is constant.

    """
    x, y = lc_pts

    # each point differs by 1/n
    dx = 1 / x.shape[0]

    if method == 'trapezoids':
        # 0 percent of the population has zero percent of the goods
        h_0 = 0.0
        h_n = y[-1]
        # the 0th entry is at x=1/n
        sum_hs = y[:-1].sum()
        return dx * ((h_0 + h_n) / 2 + sum_hs)
    elif method == 'rectangles':
        return dx * y.sum()
    else:
        raise ValueError("Method '%s' not implemented. Available methods: "
                         "'rectangles', 'trapezoids'." % method)