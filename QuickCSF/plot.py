import numpy

import matplotlib
import matplotlib.pyplot as plt

from . import QuickCSF

# qw: import our utils
from utility import utils

def plot(qCSFEstimator, graph=None, unmappedTrueParams=None, showNumbers=True, show=True, trueThresholdCurve=None, frequency_labels=[1,4,16,64]):
	'''Generate a plot of estimates from QuickCSF, along with history of responses and true parameter values'''
	
	frequencyDomain = qCSFEstimator.stimulusSpace[1].reshape(-1, 1)

	if graph is None:
		fig = plt.figure()
		graph = fig.add_subplot(1, 1, 1)

		if show:
			plt.ion()
			plt.show()

	if unmappedTrueParams is not None:
		truthData = QuickCSF.csf_unmapped(unmappedTrueParams.reshape(1, -1), frequencyDomain)
		truthData = numpy.power(10, truthData)
		truthLine = graph.fill_between(
			frequencyDomain.reshape(-1),
			truthData.reshape(-1),
			color=(1, 0, 0, .5)
		)
	else:
		if trueThresholdCurve is not None:
			# plot the spline
			left, right, _, _ = utils.get_data_bounds(trueThresholdCurve)
			cs = utils.create_cubic_spline(trueThresholdCurve)

			# get values in log10 units
			freqs = frequencyDomain.reshape(-1)
			logFreqs = numpy.log10(freqs)
			logContrasts = cs(logFreqs)
			contrasts = numpy.power(10, logContrasts)

			# plot with real units
			graph.plot(freqs, contrasts, color=(1, 0, 0, .5))

			logFreqs = numpy.linspace(left, right, 750)
			freqs = 10**logFreqs
			logContrasts = cs(logFreqs)
			contrasts = numpy.power(10, logContrasts)
			graph.plot(freqs, contrasts, color=(1, 0, 1, .5))
		
		truthData = None

	estimatedParamMeans = qCSFEstimator.getResults(leaveAsIndices=True)
	estimatedParamMeans = numpy.array([[
		estimatedParamMeans['peakSensitivity'],
		estimatedParamMeans['peakFrequency'],
		estimatedParamMeans['bandwidth'],
		estimatedParamMeans['delta'],
	]])
	estimatedData = QuickCSF.csf_unmapped(estimatedParamMeans.reshape(1, -1), frequencyDomain)
	estimatedData = numpy.power(10, estimatedData)

	estimatedLine = graph.fill_between(
		frequencyDomain.reshape(-1),
		estimatedData.reshape(-1),
		color=(0, 0, 1, .4)
	)

	## Chart responses
	positives = {'f':[], 's':[]}
	negatives = {'f':[], 's':[]}
	for record in qCSFEstimator.responseHistory:
		stimValues = record[0]
		targetArray = positives if record[1] else negatives
		targetArray['f'].append(stimValues[1])
		targetArray['s'].append(1/stimValues[0])

	graph.plot(positives['f'], positives['s'], 'o', markersize=4, color=(.2, 1, .2))
	graph.plot(negatives['f'], negatives['s'], 'x', markersize=5, color=(1,0,0), markeredgewidth=2)

	graph.set_xlabel('Spatial Frequency (cyc/deg)')
	graph.set_xscale('log')
	graph.get_xaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:g}'))

	graph.set_ylabel('Contrast')
	graph.set_yscale('log')
	graph.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


	# qw
	if unmappedTrueParams is not None:
		graph.set_xlim((.25, 64))
		graph.set_xticks([1, 2, 4, 8, 16, 32])

		graph.set_ylim((1, 400))
		graph.set_yticks([2, 10, 50, 200])
	else:
		graph.set_xlim(frequency_labels[0], frequency_labels[-1])
		graph.set_xticks(frequency_labels, format='%g')
		
		graph.set_ylim((1, 1000))
		graph.set_yticks([1, 10, 100, 1000], format='%g')
		graph.set_yticklabels([1, 0.1, 0.01, 0.001])

	graph.grid()

	if showNumbers:
		estimatedParamMeans = QuickCSF.mapCSFParams(estimatedParamMeans, exponify=True)
		estimatedParamMeans = estimatedParamMeans.reshape(1,-1).tolist()[0]
		paramEstimates = '%03.2f, %.4f, %.4f, %.4f' % tuple(estimatedParamMeans)
		estimatedLine.set_label(f'Estim: {paramEstimates}')

		if truthData is not None:
			trueParams = QuickCSF.mapCSFParams(unmappedTrueParams, True).T.tolist()[0]
			trueParams = '%03.2f, %.4f, %.4f, %.4f' % tuple(trueParams)
			truthLine.set_label(f'Truth: {trueParams}')

		graph.legend()

	if show:
		plt.pause(0.004) # necessary for non-blocking graphing

	return graph
