package com.specpal.mobileai;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class MainActivity extends AppCompatActivity {

    //Load the tensorflow inference library
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private String MODEL_PATH = "file:///android_asset/squeezenet.pb";
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[2];
    private float[] floatValues;
    private int[] INPUT_SIZE = {224,224,3};

    ImageView imageView;
    TextView resultView;
    Snackbar progressBar;

    private static final int CHOOSE_PHOTO=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);


        //initialize tensorflow with the AssetManager and the Model
        // 这里也需要改，但这个 tensorflow 的代码我不会改也改不了
        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);

        imageView = (ImageView) findViewById(R.id.imageview);
        resultView = (TextView) findViewById(R.id.results);

        progressBar = Snackbar.make(imageView,"PROCESSING IMAGE",Snackbar.LENGTH_INDEFINITE);


        final FloatingActionButton predict = (FloatingActionButton) findViewById(R.id.predict);
        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // 选择手机相册
                Intent intent = new Intent("android.intent.action.GET_CONTENT");
                intent.setType("image/*");
                startActivityForResult(intent, CHOOSE_PHOTO);

//                try{
//
//                    //READ THE IMAGE FROM ASSETS FOLDER
//                    InputStream imageStream = getAssets().open("testimage.jpg");
//
//                    Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
//
//                    imageView.setImageBitmap(bitmap);
//
//                    progressBar.show();
//
//                    predict(bitmap);
//                }
//                catch (Exception e){
//
//                }

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // 选择图片成功  回调
        switch (requestCode) {
            case CHOOSE_PHOTO:
                if (resultCode == RESULT_OK) {
                    Bitmap bitmap = null;
                    //判断手机系统版本号
                    if (Build.VERSION.SDK_INT >= 19) {
                        //4.4及以上系统使用这个方法处理图片
                        bitmap = ImgUtil.handleImageOnKitKat(this, data);        //ImgUtil是自己实现的一个工具类
                    } else {
                        //4.4以下系统使用这个方法处理图片
                        bitmap = ImgUtil.handleImageBeforeKitKat(this, data);
                    }
                    imageView.setImageBitmap(bitmap);

                    progressBar.show();

                    predict(bitmap);
                }
                break;
            default:
                break;
        }
    }


    //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
    public Object[] argmax(float[] array){


        int best = -1;
        float best_confidence = 0.0f;

        for(int i = 0;i < array.length;i++){

            float value = array[i];

            if (value > best_confidence){

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best,best_confidence};


    }


    public void predict(final Bitmap bitmap){


        //Runs inference in background thread
        new AsyncTask<Integer,Integer,Integer>(){

            @Override

            protected Integer doInBackground(Integer ...params){

                //Resize the image into 224 x 224
                Bitmap resized_image = ImageUtils.processBitmap(bitmap,224);

                //Normalize the pixels
                floatValues = ImageUtils.normalizeBitmap(resized_image,224,127.5f,1.0f);

                //Pass input into the tensorflow
                tf.feed(INPUT_NAME,floatValues,1,224,224,3);

                //compute predictions
                tf.run(new String[]{OUTPUT_NAME});

                //copy the output into the PREDICTIONS array
                tf.fetch(OUTPUT_NAME,PREDICTIONS);

                //Obtained highest prediction
                Object[] results = argmax(PREDICTIONS);


                int class_index = (Integer) results[0];
                float confidence = (Float) results[1];


                try{

                    final String conf = String.valueOf(confidence * 100).substring(0,5);

                    //Convert predicted class index into actual label name
                   final String label = ImageUtils.getLabel(getAssets().open("labels.json"),class_index);



                   //Display result on UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {

                            progressBar.dismiss();
                            resultView.setText(label + " : " + conf + "%");

                        }
                    });

                }

                catch (Exception e){


                }


                return 0;
            }



        }.execute(0);

    }


}
