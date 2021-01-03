/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tflite;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier extends AppCompatActivity {
  private static final Logger LOGGER = new Logger();

  /** The model type used for classification. */
  public enum Model {
    FLOAT_MOBILENET,
    QUANTIZED_MOBILENET,
    FLOAT_EFFICIENTNET,
    QUANTIZED_EFFICIENTNET,
    FLOAT_INCEPTION,
    QUANTIZED_INCEPTION
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    DSP,
    GPU,
    NNAPI_DSP,
    NNAPI_GPU
  }

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** Image size along the x axis. */
  private final int imageSizeX;

  /** Image size along the y axis. */
  private final int imageSizeY;

  /** Optional GPU delegate for accleration. */
  private GpuDelegate gpuDelegate = null;

  /** Optional NNAPI delegate for accleration. */
  private NnApiDelegate nnApiDelegate = null;

  private NnApiDelegate.Options nnapiOptions = null;

  /** Optional Hexagon DSP delegate for accleration. */
  private HexagonDelegate hexagonDelegate = null;

  private String delegateName = "";

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** Input image TensorBuffer. */
  private TensorImage inputImageBuffer;

  /** Output probability TensorBuffer. */
  private final TensorBuffer outputProbabilityBuffer;

  /** Processer to apply post processing of the output probability. */
  private final TensorProcessor probabilityProcessor;

  private final ArrayList<Long> capture_times = new ArrayList<Long>();
  private final ArrayList<Long> preproc_times = new ArrayList<Long>();
  private final ArrayList<Long> postproc_times = new ArrayList<Long>();
  private final ArrayList<Long> inference_times = new ArrayList<Long>();

  private final static int PERMISSION_REQUEST_CODE = 1;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Model model, int nnapiOption, Device device, int numThreads)
      throws IOException {

    if(ContextCompat.checkSelfPermission(activity,
            Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED)
    {
      LOGGER.d("REQUESTING PERMISSION");
      ActivityCompat.requestPermissions(activity,
              new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
              1);
      LOGGER.d("REQUESTED PERMISSION");
    }

    if (model == Model.QUANTIZED_MOBILENET) {
      return new ClassifierQuantizedMobileNet(activity, nnapiOption, device, numThreads);
    } else if (model == Model.FLOAT_MOBILENET) {
      return new ClassifierFloatMobileNet(activity, nnapiOption, device, numThreads);
    } else if (model == Model.FLOAT_EFFICIENTNET) {
      return new ClassifierFloatEfficientNet(activity, nnapiOption, device, numThreads);
    } else if (model == Model.QUANTIZED_EFFICIENTNET) {
      return new ClassifierQuantizedEfficientNet(activity, nnapiOption, device, numThreads);
    } else if (model == Model.FLOAT_INCEPTION) {
      return new ClassifierFloatInception(activity, nnapiOption, device, numThreads);
    } else if (model == Model.QUANTIZED_INCEPTION) {
      return new ClassifierQuantizedInception(activity, nnapiOption, device, numThreads);
    } else {
      throw new UnsupportedOperationException();
    }
  }

  private double calculateAverage(List <Long> marks) {
    Double sum = 0D;
    if(!marks.isEmpty()) {
      for (Long mark : marks) {
        sum += mark;
      }
      return sum.doubleValue() / marks.size();
    }
    return sum;
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  private String enum2acceleratorName(Device device)
  {
    switch (device)
    {
      case NNAPI_DSP: return "qti-dsp";
      case NNAPI_GPU: return "qti-gpu";
      case NNAPI: return "nnapi";
      case GPU: return "gpu";
      case DSP: return "dsp";
      case CPU: return "cpu";
    }
    return "";
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, int nnapiOption, Device device, int numThreads) throws IOException {
    tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
    nnapiOptions = new NnApiDelegate.Options();
    nnapiOptions.setExecutionPreference(nnapiOption);

    switch (device) {
      case NNAPI_DSP:
      case NNAPI_GPU:
        nnapiOptions.setAcceleratorName(enum2acceleratorName(device));
      case NNAPI:
        LOGGER.d("Creating a delegate with preference: " + nnapiOption);
        nnApiDelegate = new NnApiDelegate(nnapiOptions);
        tfliteOptions.addDelegate(nnApiDelegate);
        delegateName = "nnapi";
        break;
      case GPU:
        gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);
        delegateName = "gpu";
        break;
      case DSP:
        hexagonDelegate = new HexagonDelegate(activity);
        tfliteOptions.addDelegate(hexagonDelegate);
        delegateName = "hexagon";
        break;
      case CPU:
        delegateName = "cpu";
        break;
    }
    tfliteOptions.setNumThreads(numThreads);
    tflite = new Interpreter(tfliteModel, tfliteOptions);

    // Loads labels out from the label file.
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

    LOGGER.d("Created a Tensorflow Lite Image Classifier.");
  }

  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(final Bitmap bitmap, int sensorOrientation, long data) throws IOException {
    // LOGGER.d(String.format("%d", preproc_times.size()));
    // Logs this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    // Trace.beginSection("loadImage");
    long startTimeForLoadImage = SystemClock.uptimeMillis();
    inputImageBuffer = loadImage(bitmap, sensorOrientation);
    long endTimeForLoadImage = SystemClock.uptimeMillis();
    // Trace.endSection();
    // LOGGER.d("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

    // Runs the inference call.
    Trace.beginSection("runInference");
    long startTimeForReference = SystemClock.uptimeMillis();
    tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
    long endTimeForReference = SystemClock.uptimeMillis();
    Trace.endSection();
    // LOGGER.d("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

    Trace.beginSection("postProcess");
    long startTimeForPostproc = SystemClock.uptimeMillis();
    // Gets the map of label and probability.
    Map<String, Float> labeledProbability =
        new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
            .getMapWithFloatValue();

    // Gets top-k results.
    List<Recognition> topK = getTopKProbability(labeledProbability);
    long endTimeForPostproc = SystemClock.uptimeMillis();
    Trace.endSection(); // postProcess

    Trace.endSection(); // recognizeImage

    capture_times.add(data);
    preproc_times.add(endTimeForLoadImage - startTimeForLoadImage);
    postproc_times.add(endTimeForPostproc - startTimeForPostproc);
    inference_times.add(endTimeForReference - startTimeForReference);

    //if(preproc_times.size() == 100) {
    //  String cost_str = String.format("Timecosts: %f %f %f %f", calculateAverage(capture_times), calculateAverage(preproc_times), calculateAverage(inference_times), calculateAverage(postproc_times));
    //  LOGGER.d(cost_str);
//
    //  capture_times.clear();
    //  preproc_times.clear();
    //  postproc_times.clear();
    //  inference_times.clear();
//
    //  System.exit(0);
    //}

    //if(preproc_times.size() == 1200)
    //{
    //  String filename = "/sdcard/app_distributions/classify/" + this.getModelPath().toLowerCase() + "_" + this.delegateName + ".csv";
    //  LOGGER.d("Saving to " + filename);
    //  BufferedWriter br = new BufferedWriter(new FileWriter(filename));
    //  StringBuilder sb = new StringBuilder();
//
    //  // Append strings from array
    //  for (int i = 0; i < preproc_times.size(); ++i) {
    //    sb.append(Long.toString(capture_times.get(i)));
    //    sb.append(",");
    //    sb.append(Long.toString(preproc_times.get(i)));
    //    sb.append(",");
    //    sb.append(Long.toString(inference_times.get(i)));
    //    sb.append(",");
    //    sb.append(Long.toString(postproc_times.get(i)));
    //    sb.append("\n");
    //  }
//
    //  br.write(sb.toString());
    //  br.close();
    //  System.exit(0);
    //}

    // LOGGER.d("Timecost to post-process (i.e., select topK) results: " + (endTimeForPostproc - startTimeForPostproc));
    return topK;
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
    if (hexagonDelegate != null) {
      hexagonDelegate.close();
      hexagonDelegate = null;
    }
    tfliteModel = null;
  }

  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return imageSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return imageSizeY;
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    // Loads bitmap into a TensorImage.
    Trace.beginSection("loadImage");
    inputImageBuffer.load(bitmap);
    Trace.endSection();

    Trace.beginSection("preProcess");
    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRotation = sensorOrientation / 90;
    // TODO(b/143564309): Fuse ops inside ImageProcessor.
    if(true) {
      ImageProcessor imageProcessor =
              new ImageProcessor.Builder()
                      .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                      .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                      .add(new Rot90Op(numRotation))
                      .add(getPreprocessNormalizeOp())
                      .build();

      TensorImage ret = imageProcessor.process(inputImageBuffer);

      Trace.endSection();
      return ret;
    } else { // BELOW IS FOR TRACING
      ImageProcessor imageProcessor = new ImageProcessor.Builder().add(new ResizeWithCropOrPadOp(cropSize, cropSize)).build();
      Trace.beginSection("crop/pad");
      TensorImage ret = imageProcessor.process(inputImageBuffer);
      Trace.endSection();

      imageProcessor = new ImageProcessor.Builder().add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR)).build();
      Trace.beginSection("resize");
      ret = imageProcessor.process(ret);
      Trace.endSection();

      imageProcessor = new ImageProcessor.Builder().add(new Rot90Op(numRotation)).build();
      Trace.beginSection("rotate");
      ret = imageProcessor.process(ret);
      Trace.endSection();

      imageProcessor = new ImageProcessor.Builder().add(getPreprocessNormalizeOp()).build();
      Trace.beginSection("normalize");
      ret = imageProcessor.process(ret);
      Trace.endSection();

      Trace.endSection();
      return ret;
    }
  }

  /** Gets the top-k results. */
  private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
    // Find the best classifications.
    PriorityQueue<Recognition> pq =
        new PriorityQueue<>(
            MAX_RESULTS,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
      pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
    }

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    return recognitions;
  }

  /** Gets the name of the model file stored in Assets. */
  protected abstract String getModelPath();

  /** Gets the name of the label file stored in Assets. */
  protected abstract String getLabelPath();

  /** Gets the TensorOperator to nomalize the input image in preprocessing. */
  protected abstract TensorOperator getPreprocessNormalizeOp();

  /**
   * Gets the TensorOperator to dequantize the output probability in post processing.
   *
   * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
   * essentially linear transformation). For float model, de-quantize is not required. But to
   * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
   * 1.0f, respectively.
   */
  protected abstract TensorOperator getPostprocessNormalizeOp();
}
