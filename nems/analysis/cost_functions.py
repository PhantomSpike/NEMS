import logging

import nems.utils
import nems.fitters.util

log = logging.getLogger(__name__)


def basic_cost(sigma, unpacker, modelspec, data, segmentor,
               evaluator, metric, keep_stack=False):
    '''Standard cost function for use by fit_basic and other analyses.'''
    updated_spec = unpacker(sigma)
    # The segmentor takes a subset of the data for fitting each step
    # Intended use is for CV or random selection of chunks of the data
    # For fit_basic the 'segmentor' just passes it all through.
    data_subset = segmentor(data)

    start_idx = None
    if keep_stack and hasattr(basic_cost, 'stack'):
        idx = nems.fitters.util.last_matching_phi(modelspec, updated_spec)
        data_subset = nems.fitters.util.copy_stack_data(
                                data_subset, basic_cost.stack, idx,
                                modelspec
                                )
        start_idx = idx+1

    updated_data_subset = evaluator(data_subset, updated_spec,
                                    start=start_idx, keep_stack=keep_stack)

    if keep_stack:
        basic_cost.stack = updated_data_subset['stack']

    error = metric(updated_data_subset)
    log.debug("inside cost function, current error: %.06f", error)
    log.debug("current sigma: %s", sigma)

    if hasattr(basic_cost, 'counter'):
        basic_cost.counter += 1
        if basic_cost.counter % 500 == 0:
            log.info('Eval #%d. E=%.06f', basic_cost.counter, error)
            nems.utils.progress_fun()

    if hasattr(basic_cost, 'error'):
        basic_cost.error = error

    return error
