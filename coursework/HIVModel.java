package dl4j.hiv.hivmodel;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import static java.lang.System.out;

public class HIVModel {

	private static final int FIELDS_COUNT = 4;
	private static final int CLASSIFICATIONS = 3;
	private static final int NUM_RECORDS = 273;

	public static void main(String[] args) {

		try (RecordReader reader = new CSVRecordReader(0, ',')) {
			reader.initialize(new FileSplit(
					new File("C:\\Eclipse Projects\\hivmodel\\hiv_complete_data_set.csv")));

			DataSetIterator iter = new RecordReaderDataSetIterator(
					reader, NUM_RECORDS, FIELDS_COUNT, CLASSIFICATIONS);
			DataSet completeData = iter.next();
			//completeData.shuffle(System.currentTimeMillis());
			completeData.shuffle(42);

			DataNormalization norm = new NormalizerStandardize();
			norm.fit(completeData);
			norm.transform(completeData);

			SplitTestAndTrain splitData = completeData.splitTestAndTrain(0.65);
			DataSet trainData = splitData.getTrain();
			DataSet testData = splitData.getTest();

			MultiLayerConfiguration config 
			= new NeuralNetConfiguration.Builder()
			.iterations(1000)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.activation(Activation.SELU)
			.weightInit(WeightInit.XAVIER)
			.learningRate(0.1)
			.regularization(true).l2(0.0001)
			.list()
			.layer(0, new DenseLayer.Builder().nIn(FIELDS_COUNT).nOut(3).build())
			.layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
			.layer(2, new OutputLayer.Builder(
					LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					.activation(Activation.SOFTMAX)
					.nIn(3).nOut(CLASSIFICATIONS).build())
			.backprop(true).pretrain(false)
			.build();

			MultiLayerNetwork model = new MultiLayerNetwork(config);
			model.init();
			model.fit(trainData);

			INDArray output = model.output(testData.getFeatureMatrix());
			Evaluation eval = new Evaluation(CLASSIFICATIONS);
			eval.eval(testData.getLabels(), output);

			//out.println(output);
			out.println(eval.stats());


		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
