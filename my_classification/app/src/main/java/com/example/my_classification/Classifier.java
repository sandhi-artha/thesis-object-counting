package com.example.my_classification;
import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

// first add into build.gradle (Module:app): implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

// file utilities
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// adding basic image processing
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public abstract class Classifier {
    public enum Model {
        FLOAT, QUANTIZED
    }

    public enum Device {
        CPU, NNAPI, GPU
    }

    // Device options
    private GpuDelegate gpuDelegate = null;
    private NnApiDelegate nnApiDelegate = null;


    // TFLITE SPECIFIC VARIABLES
    // declare interpreter object to run inference with tflite
    private Interpreter tflite;
    // load model's labels
    private List<String> labels;
    // Set options for tflite interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // declare the loaded tflite model as bytebuffer type
    private MappedByteBuffer tfliteModel;

    // IMAGE VARIABLES
    private final int imageSizeX;
    private final int imageSizeY;
    // input image tensorBuffer
    private TensorImage inputImageBuffer;
    // Output probability TensorBuffer (final)
    private final TensorBuffer outputProbabilityBuffer;
    // Processer to apply post processing of the output probability (final)
    private final TensorProcessor probabilityProcessor;

    /** 1. creates classifier object with current activity, model etc*/
    public static Classifier create(Activity activity, Model model, Device device, int numThreads)
            throws IOException {
        if (model == Model.QUANTIZED) {
            return new ClassifierQuantizedMobileNet(activity, device, numThreads);
        } else {
            return new ClassifierFloatMobileNet(activity, device, numThreads);
        }
    }

    // An immutable result returned by a Classifier describing what was recognized.
    public static class Recognition{
        private final String title;
        private final Float count;

        public Recognition(String title, float count) {
            this.title = title;
            this.count = count;
        }
        public String getTitle() {
            return title;
        }
        public Float getCount() {
            return count;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (count != null) {
                resultString += String.format("%.2f ", count);
            }
            if (title != null) {
                resultString += title + " ";
            }
            return resultString.trim();
        }
    }



    // INIT FUNCTION - load the model and label and catch IO exception
    protected Classifier(Activity activity, Device device, int numThreads) throws IOException{
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        switch (device) {
            case NNAPI:
                nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(1);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        Log.d("Classifier", "Loaded model successfully");

        // load the labels
        labels = FileUtil.loadLabels(activity, getLabelPath());

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
        Log.d("Classifier","Created a Tensorflow Lite Image Classifier");
    }

    /** 2. run recognize image using the converted bitmap drawable as input*/
    // Runs inference and returns the classification results
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("loadImage");
        long startTimeForLoadImage = SystemClock.uptimeMillis();

        /** 3. loads the image resize and do preprocess normalization*/
        inputImageBuffer = loadImage(bitmap);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.d("Classifier", "Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();

        /** 4. runs inference and store the results in outputProbabilityBuffer*/
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.d("Classifier","Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        // Gets the map of label and probability.
        /** 5. map the output probability with the labels, it should be the same size
         * labeledProbability will have the category name as first value, and its prediction in second value*/
        Map<String, Float> labeledProbability =
            new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                    .getMapWithFloatValue();

        Trace.endSection();

        // Gets top-k results.
        return getFilteredCount(labeledProbability);
    }

    /** get count above threshold value */
    private static List<Recognition> getFilteredCount(Map<String, Float> labelProb){
        float thresh = 0.8f;
        ArrayList<Recognition> filterCount = new ArrayList<>();
        for (Map.Entry<String, Float> entry : labelProb.entrySet()){
            if(entry.getValue()>thresh){
                filterCount.add(new Recognition(entry.getKey(), entry.getValue()));
            }
        }
        return filterCount;

    }

    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        tfliteModel = null;
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreprocessNormalizeOp())
                .build();
        return imageProcessor.process(inputImageBuffer);
    }






    /** Gets the name of the model file stored in Assets. */
    protected abstract String getModelPath();

    /** Gets the name of the label file stored in Assets. */
    protected abstract String getLabelPath();

    /** Gets the TensorOperator to nomalize the input image in preprocessing. */
    protected abstract TensorOperator getPreprocessNormalizeOp();
    protected abstract TensorOperator getPostprocessNormalizeOp();








}
