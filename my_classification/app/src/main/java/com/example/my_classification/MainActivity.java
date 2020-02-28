package com.example.my_classification;

// for handling bitmap data
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.BitmapDrawable;

import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private Button btnPred;
    private Button btnLoad;

    private long lastProcessingTimeMs;

    private Classifier classifier;
    private int imgCount = 0;
    private TextView textPred;
    private ImageView imgView;
    private int[] img_array = {R.drawable.test1, R.drawable.test2, R.drawable.test3,
            R.drawable.test4, R.drawable.test5, R.drawable.test6, R.drawable.test7,
            R.drawable.test8, R.drawable.test9, R.drawable.test10, R.drawable.test11,
            R.drawable.test12, R.drawable.test13, R.drawable.test14, R.drawable.test15};

    private TextView inferenceTimeTextView;

    // BITMAP CONVERSION
    Bitmap bm;
    Drawable dw;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // create handles
        btnPred = findViewById(R.id.btnPred);
        btnLoad = findViewById(R.id.btnLoad);
        textPred = findViewById(R.id.textPrediction);
        imgView = findViewById(R.id.imageView);
        inferenceTimeTextView = findViewById(R.id.inference_info);

        try {
            classifier = Classifier.create(this, Classifier.Model.FLOAT, Classifier.Device.CPU, 1);
        }catch(IOException e) {
            Log.e("activity", "Failed to create classifier.");
        }


        // debug a click
        btnPred.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d("activity", "onClick: Doing Predictions");
                dw = imgView.getDrawable();

                bm = ((BitmapDrawable)dw).getBitmap();
                Log.d("activity", "converted to bitmap");
                final long startTime = SystemClock.uptimeMillis();
                List<Classifier.Recognition> result = classifier.recognizeImage(bm);
                Log.d("activity", "obtained results");
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                showInference(lastProcessingTimeMs + "ms");

                if(result != null){
                    textPred.setText(result.toString());
                }else{
                    textPred.setText("No predictions!");
                }
            }
        });

        btnLoad.setOnClickListener((new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imgView.setImageResource(img_array[imgCount]);
                imgCount = (imgCount + 1) % img_array.length;
                textPred.setText("Hit predict button!");
            }
        }));


    }

    protected void showInference(String inferenceTime) {
        inferenceTimeTextView.setText(inferenceTime);
    }

}
