from scipy.stats import spearmanr, pearsonr, kendalltau

def spearman_corr(output, target):
    '''
    Calculate a Spearman correlation coefficient with associated p-value.
    More info at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html#scipy.stats.spearmanr
    :param output: model predictions.
    :param target: target results.
    :return: (correlation, pvalue)
    '''
    correlation, pvalue = spearmanr(output, target, axis=None)
    return correlation, pvalue

def pearson_corr(output, target):
    '''
    Pearson correlation coefficient and p-value for testing non-correlation.
    More info at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    :param output: model predictions.
    :param target: target results.
    :return: (correlation, pvalue)
    '''
    correlation, pvalue = pearsonr(output, target)
    return correlation, pvalue

def kendalltau_corr(output, target):
    '''
    Calculate Kendall’s tau, a correlation measure for ordinal data.
    More info at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html#scipy.stats.kendalltau
    :param output: model predictions.
    :param target: target results.
    :return: (correlation, pvalue)
    '''
    correlation, pvalue = kendalltau(output, target)
    return correlation, pvalue