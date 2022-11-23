from trainer import metric as module_metric

class MetricHandler:
    def __init__(self, metrics_names):
        self._metric_functions = [getattr(module_metric, met) for met in metrics_names]
        self._data = dict.fromkeys(metrics_names, 0)
        self._data_with_pvalue = dict.fromkeys(metrics_names, 0)
        self.reset()

    def reset(self):
        for keys in self._data:
            self._data[keys] = 0
        for keys in self._data_with_pvalue:
            self._data_with_pvalue[keys] = 0

    def add(self, key, value):
        self._data[key] = value
        self._data_with_pvalue[key] = value

    def update(self, output, target):
        for fn in self._metric_functions:
            self._data[fn.__name__] = fn(output, target)[0]
            self._data_with_pvalue[fn.__name__] = fn(output, target)

    def get_data(self):
        return self._data.copy()

    def get_data_with_pvalue(self):
        return self._data_with_pvalue.copy()