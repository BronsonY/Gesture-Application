package com.example.mygestureappv3

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.os.FileUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.AutoCompleteTextView
import android.widget.TextView
import androidx.core.app.ActivityCompat
import com.example.mygestureappv3.ml.LatestGestureModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer



class MainActivity : AppCompatActivity() {


    lateinit var labels: List<String>
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap: Bitmap
    lateinit var capReq: CaptureRequest.Builder
    lateinit var handler: Handler
    lateinit var handlerThread: HandlerThread
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var cameraCaptureSession: CameraCaptureSession
    lateinit var cameraDevice: CameraDevice
    lateinit var captureRequest: CaptureRequest
    lateinit var model: LatestGestureModel
    lateinit var textView: TextView

    private val backgroundHandler: Handler by lazy {
        val handlerThread = HandlerThread("backgroundThread")
        handlerThread.start()
        Handler(handlerThread.looper)
    }

    private var framesCaptured = 0

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permissions()

        labels = FileUtil.loadLabels(this, "labels.txt")
        // Initialize your input shape
        val inputShape = intArrayOf(1, 36, 100, 100, 1)

        // Create an empty FloatArray to store the image data
        val imagedata = FloatArray(inputShape[1] * inputShape[2] * inputShape[3] * inputShape[4])

        imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(100, 100, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = LatestGestureModel.newInstance(this)
        textureView = findViewById(R.id.textureView)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler((handlerThread).looper)
        textView = findViewById(R.id.text_view)

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            val imagedata = FloatArray(36 * 100 * 100 * 1)
            var currentFrameIndex = 0
            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                Log.d("SurfaceTexture", "Surface Texture Available")
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                // Handle size changes if needed
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                // Handle texture destroyed event if needed
                return true
            }

            @SuppressLint("SetTextI18n")
            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                val originalBitmap = textureView.bitmap!!

                // Convert the Bitmap to grayscale and resize it to 100x100
                val grayscaleBitmap = convertToGrayscaleAndResize(originalBitmap, 100, 100)

                // Convert the grayscaleBitmap to a FloatArray and normalize pixel values
                val framePixels = FloatArray(100 * 100 * 1)
                val buffer = ByteBuffer.allocate(4 * framePixels.size)
                grayscaleBitmap.copyPixelsToBuffer(buffer)
                buffer.rewind()

                for (i in framePixels.indices) {
                    framePixels[i] = (buffer.float / 255.0).toFloat()  // Normalize pixel values to [0, 1]
                }

                // Check for NaN values in the framePixels
                if (framePixels.any { it.isNaN() }) {
                    Log.e("NaNCheck", "Found NaN values in framePixels:")
                    framePixels.forEachIndexed { index, value ->
                        if (value.isNaN()) {
                            Log.e("NaNCheck", "NaN found at index $index")
                        }
                    }
                    return
                }

                // Add the framePixels to the imagedata at the current frame index
                System.arraycopy(framePixels, 0, imagedata, currentFrameIndex * framePixels.size, framePixels.size)
                currentFrameIndex++

                if (currentFrameIndex >= 36) {

                    // Define the desired input shape
                    val input_Shape = intArrayOf(1, 36, 100, 100, 1)

                    val reshapedImageData = Array(input_Shape[0]) {
                        Array(input_Shape[1]) {
                            Array(input_Shape[2]) {
                                Array(input_Shape[3]) {
                                    FloatArray(input_Shape[4])
                                }
                            }
                        }
                    }

                    for (i in 0 until input_Shape[0]) {
                        for (j in 0 until input_Shape[1]) {
                            for (k in 0 until input_Shape[2]) {
                                for (l in 0 until input_Shape[3]) {
                                    for (m in 0 until input_Shape[4]) {
                                        val index = i * (input_Shape[1] * input_Shape[2] * input_Shape[3] * input_Shape[4]) +
                                                j * (input_Shape[2] * input_Shape[3] * input_Shape[4]) +
                                                k * (input_Shape[3] * input_Shape[4]) +
                                                l * input_Shape[4] +
                                                m
                                        reshapedImageData[i][j][k][l][m] = imagedata[index]
                                    }
                                }
                            }
                        }
                    }

                    // Flatten the reshapedImageData to a 1D array
                    val flattenedImageData = reshapedImageData.flatMap { array1 ->
                        array1.flatMap { array2 ->
                            array2.flatMap { array3 ->
                                array3.flatMap { it.asList() }
                            }
                        }
                    }.toFloatArray()

                    val inputFeature0 = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)

                    inputFeature0.loadArray(flattenedImageData)

                    // Log the input data after loading it into TensorBuffer
                    val inputShape = inputFeature0.shape
                    Log.d("InputShape", "Input Shape: ${inputShape.contentToString()}")

                    // Log the input data after loading it into TensorBuffer
                    val loadedData = inputFeature0.floatArray
                    Log.d("LoadedData", "Loaded Data: ${loadedData.joinToString(", ")}")

                    // Run model inference
                    val outputs = model.process(inputFeature0)
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                    // Assuming outputFeature0 contains floating-point values
                    val outputValues = outputFeature0.floatArray

                    val argmaxIndex = argmax(outputValues)
                    Log.d("Result", "Result Values: $argmaxIndex")

                    val predictedLabel = labels[argmaxIndex]

                    runOnUiThread {
                        textView.text = "Predicted: $predictedLabel"
                    }

                    val outputShape = outputValues.size
                    Log.d("OutputShape", "Output Shape: $outputShape")

                    val outputValuesString = outputValues.joinToString(", ")
                    Log.d("output_value", "Output Values: $outputValuesString")

                    // Reset the frame index to 0 after processing
                    currentFrameIndex = 0
                }
            }
        }
    }

    fun open_camera(){
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        cameraManager.openCamera(cameraManager.cameraIdList[0], object: CameraDevice.StateCallback(){
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                val capReq = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                val surface = Surface(textureView.surfaceTexture)
                capReq.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object:
                    CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        cameraCaptureSession.setRepeatingRequest(capReq.build(),null,null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {

                    }
                },handler)
            }

            override fun onDisconnected(camera: CameraDevice) {

            }

            override fun onError(camera: CameraDevice, error: Int) {

            }
        },handler)
    }

    fun get_permissions(){
        val permissionsLst = mutableListOf<String>()

        if(checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED)
            permissionsLst.add(Manifest.permission.CAMERA)
        if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED)
            permissionsLst.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        if(checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED)
            permissionsLst.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)

        if(permissionsLst.size>0){
            requestPermissions(permissionsLst.toTypedArray(),101)
        }
    }

    fun convertToGrayscaleAndResize(src: Bitmap, desiredWidth: Int, desiredHeight: Int): Bitmap {
        val originalWidth = src.width
        val originalHeight = src.height

        val scaledBitmap = Bitmap.createScaledBitmap(src, desiredWidth, desiredHeight, false)
        val grayscaleBitmap = scaledBitmap.copy(scaledBitmap.config, true)

        for (x in 0 until desiredWidth) {
            for (y in 0 until desiredHeight) {
                val pixel = scaledBitmap.getPixel(x, y)
                val red = Color.red(pixel)
                val green = Color.green(pixel)
                val blue = Color.blue(pixel)

                val grayscaleValue = (red + green + blue) / 3
                grayscaleBitmap.setPixel(x, y, Color.rgb(grayscaleValue, grayscaleValue, grayscaleValue))

            }
        }

        for (x in 0 until desiredWidth) {
            for (y in 0 until desiredHeight) {
                val pixel = Color.red(grayscaleBitmap.getPixel(x, y))
                val normalizedValue = pixel / 255.0
                grayscaleBitmap.setPixel(x, y, Color.rgb((normalizedValue * 255).toInt(), (normalizedValue * 255).toInt(), (normalizedValue * 255).toInt()))
            }
        }

        return grayscaleBitmap
    }

    fun checkOriginalFrameRange(bitmap: Bitmap) {
        val width = bitmap.width
        val height = bitmap.height

        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixelValue = Color.red(bitmap.getPixel(x, y))

                if (pixelValue < 0 || pixelValue > 255) {
                    Log.d("FrameRangeCheck", "Pixel value out of range: $pixelValue at x=$x, y=$y")
                }
            }
        }
    }

    fun argmax(outputValues: FloatArray): Int {
        var maxIndex = 0
        var maxValue = outputValues[0]

        for (i in 1 until outputValues.size) {
            if (outputValues[i] > maxValue) {
                maxIndex = i
                maxValue = outputValues[i]
            }
        }

        return maxIndex
    }

    // used this function to save the frames
//    private fun saveFrameAsImage(bitmap: Bitmap, fileName: String) {
//        val directoryPath = getExternalFilesDir(Environment.DIRECTORY_PICTURES) // Change this to your desired directory
//        val file = File(directoryPath, fileName)
//
//        try {
//            val fos = FileOutputStream(file)
//            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
//            fos.flush()
//            fos.close()
//            Log.d("FrameSaved", "Frame saved as $fileName")
//        } catch (e: Exception) {
//            Log.e("FrameSaveError", "Error saving frame as $fileName: ${e.message}")
//        }
//    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        grantResults.forEach {
            if(it != PackageManager.PERMISSION_GRANTED){
                get_permissions()
            }
        }
    }
}


