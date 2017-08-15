
%TASK: Handwriting Recognition using K Nearest Neighbour Algorithm
%Loading the characters files in jpg format

im1=imgfts('a.jpg');
im2=imgfts('d.jpg');
im3=imgfts('m.jpg');
im4=imgfts('n.jpg');
im5=imgfts('o.jpg');
im6=imgfts('p.jpg');
im7=imgfts('q.jpg');
im8=imgfts('r.jpg');
im9=imgfts('u.jpg');
im10=imgfts('w.jpg');
features=[im1;im2;im3;im4;im5;im6;im7;im8;im9;im10];
class=[1*ones(80,1);2*ones(80,1);3*ones(80,1);4*ones(80,1);5*ones(80,1);6*ones(80,1);7*ones(80,1);8*ones(80,1);9*ones(80,1);10*ones(80,1)];
D=dist2(features,features);
imagesc(D);
[D_sorted, D_index] = sort (D,2);
Predicted_Classes=class(D_index(:,2));



trainingAccuracymat=[class,Predicted_Classes];
trainingAccuracy=class==Predicted_Classes;

%Iteraring the above procedure for the test set

testclass=[1*ones(7,1);2*ones(7,1);3*ones(7,1);4*ones(7,1);5*ones(7,1);6*ones(7,1);7*ones(7,1);8*ones(7,1);9*ones(7,1);10*ones(7,1)];

TestFeatures=imgfts('test.jpg');
T=dist2(TestFeatures,features);
[T_sorted,T_index]=sort(T,2);
PredictedTesClass=class(T_index(:,1));


load Reorder

Truetestclass=testclass(ReorderIndex);
clear ReorderIndex


testaccuracymat=[Truetestclass,PredictedTesClass];

testaccuracy=Truetestclass==PredictedTesClass;

%Showing the characters 

showcharlabels ('test.jpg', PredictedTesClass, [1 2 3 4 5 6 7 8 9 10])

%Making Confusion matrix 

[ConfusionMat1,labels] = confusionmat(Truetestclass,PredictedTesClass);

colfeature1=features(:,1);
normcol1=(colfeature1-mean(colfeature1))/std(colfeature1);


colfeature2=features(:,2);
normcol2=(colfeature2-mean(colfeature2))/std(colfeature2);


colfeature3=features(:,3);
normcol3=(colfeature3-mean(colfeature3))/std(colfeature3);


colfeature4=features(:,4);
normcol4=(colfeature4-mean(colfeature4))/std(colfeature4);

colfeature5=features(:,5);
normcol5=(colfeature5-mean(colfeature5))/std(colfeature5);


colfeature6=features(:,6);
normcol6=(colfeature6-mean(colfeature6))/std(colfeature6);


normalizedfeatures=[normcol1,normcol2,normcol3,normcol4,normcol5,normcol6];

% Measuring the distance
DN=dist2(normalizedfeatures,normalizedfeatures);
imagesc(DN);
[DN_sorted, DN_index] = sort (DN,2);
Predicted_Classes=class(DN_index(:,2));


normtrainingaccuracymatk1=[class,Predicted_Classes];
normtrainingaccuracyk1=class==Predicted_Classes;


%Importing test characters

normTestFeatures=imgfts('test.jpg');

Testcol1=normTestFeatures(:,1);
normTestcol1=(Testcol1-mean(colfeature1))/std(colfeature1);


Testcol2=normTestFeatures(:,2);
normTestcol2=(Testcol2-mean(colfeature2))/std(colfeature2);


Testcol3=normTestFeatures(:,3);
normTestcol3=(Testcol3-mean(colfeature3))/std(colfeature3);

Testcol4=normTestFeatures(:,4);
normTestcol4=(Testcol4-mean(colfeature4))/std(colfeature4);


Testcol5=normTestFeatures(:,5);
normTestcol5=(Testcol5-mean(colfeature5))/std(colfeature5);


Testcol6=normTestFeatures(:,6);
normTestcol6=(Testcol6-mean(colfeature6))/std(colfeature6);

Testnormalizedfeatures=[normTestcol1,normTestcol2,normTestcol3,normTestcol4,normTestcol5,normTestcol6];

TN=dist2(Testnormalizedfeatures,normalizedfeatures);
[TN_sorted,TN_index]=sort(TN,2);

normPredictedTesClass=class(TN_index(:,1));

normtestaccuracymatk1=[Truetestclass,normPredictedTesClass];
normtestaccuracyk1=Truetestclass==normPredictedTesClass;

showcharlabels ('test.jpg', normPredictedTesClass, [1 2 3 4 5 6 7 8 9 10])
[ConfusionMat2,labels] = confusionmat(Truetestclass,normPredictedTesClass);
% For k=5
Predicted_Classesk5=class(DN_index(:,2:6));

model=mode(Predicted_Classesk5,2);
normtrainingaccuracymatk5=[class,model];

normtrainingaccuracyk5=class==model;

normPredictedTesClass=class(TN_index(:,1:5));
modelk5=mode(normPredictedTesClass,2);

normtestaccuracymatk5=[Truetestclass,modelk5];

normtestaccuracyk5=Truetestclass==modelk5;

showcharlabels ('test.jpg', modelk5, [1 2 3 4 5 6 7 8 9 10])

[ConfusionMat3,labels] = confusionmat(Truetestclass,modelk5);



im1=imgfts2('a.jpg');
im2=imgfts2('d.jpg');
im3=imgfts2('m.jpg');
im4=imgfts2('n.jpg');
im5=imgfts2('o.jpg');
im6=imgfts2('p.jpg');
im7=imgfts2('q.jpg');
im8=imgfts2('r.jpg');
im9=imgfts2('u.jpg');
im10=imgfts2('w.jpg');


features2=[im1;im2;im3;im4;im5;im6;im7;im8;im9;im10];
class=[1*ones(80,1);2*ones(80,1);3*ones(80,1);4*ones(80,1);5*ones(80,1);6*ones(80,1);7*ones(80,1);8*ones(80,1);9*ones(80,1);10*ones(80,1)];
% Normalizing again 
colfeature11=features2(:,1);
normcol11=(colfeature11-mean(colfeature11))/std(colfeature11);


colfeature22=features2(:,2);
normcol22=(colfeature22-mean(colfeature22))/std(colfeature22);


colfeature33=features2(:,3);
normcol33=(colfeature33-mean(colfeature33))/std(colfeature33);


colfeature44=features2(:,4);
normcol44=(colfeature44-mean(colfeature44))/std(colfeature44);

colfeature55=features2(:,5);
normcol55=(colfeature55-mean(colfeature55))/std(colfeature55);


colfeature66=features2(:,6);
normcol66=(colfeature66-mean(colfeature66))/std(colfeature66);

colfeature77=features2(:,7);
normcol77=(colfeature77-mean(colfeature77))/std(colfeature77);

colfeature88=features2(:,8);
normcol88=(colfeature88-mean(colfeature88))/std(colfeature88);

colfeature99=features2(:,9);
normcol99=(colfeature99-mean(colfeature99))/std(colfeature99);

colfeature1010=features2(:,10);
normcol1010=(colfeature1010-mean(colfeature1010))/std(colfeature1010);

colfeature1111=features2(:,11);
normcol1111=(colfeature1111-mean(colfeature1111))/std(colfeature1111);

colfeature1212=features2(:,12);
normcol1212=(colfeature1212-mean(colfeature1212))/std(colfeature1212);

colfeature1313=features2(:,13);
normcol1313=(colfeature1313-mean(colfeature1313))/std(colfeature1313);


normalizedfeatures1=[normcol11,normcol22,normcol33,normcol44,normcol55,normcol66,normcol77,normcol88,normcol99,normcol1010,normcol1111,normcol1212,normcol1313];

normTestFeatures1=imgfts2('test.jpg');

Testcol11=normTestFeatures1(:,1);
normTestcol11=(Testcol11-mean(colfeature11))/std(colfeature11);


Testcol22=normTestFeatures1(:,2);
normTestcol22=(Testcol22-mean(colfeature22))/std(colfeature22);


Testcol33=normTestFeatures1(:,3);
normTestcol33=(Testcol33-mean(colfeature33))/std(colfeature33);

Testcol44=normTestFeatures1(:,4);
normTestcol44=(Testcol44-mean(colfeature44))/std(colfeature44);


Testcol55=normTestFeatures1(:,5);
normTestcol55=(Testcol55-mean(colfeature55))/std(colfeature55);


Testcol66=normTestFeatures1(:,6);
normTestcol66=(Testcol66-mean(colfeature66))/std(colfeature66);

Testcol77=normTestFeatures1(:,7);
normTestcol77=(Testcol77-mean(colfeature77))/std(colfeature77);

Testcol88=normTestFeatures1(:,8);
normTestcol88=(Testcol88-mean(colfeature88))/std(colfeature88);

Testcol99=normTestFeatures1(:,9);
normTestcol99=(Testcol99-mean(colfeature99))/std(colfeature99);

Testcol1010=normTestFeatures1(:,10);
normTestcol1010=(Testcol1010-mean(colfeature1010))/std(colfeature1010);

Testcol1111=normTestFeatures1(:,11);
normTestcol1111=(Testcol1111-mean(colfeature1111))/std(colfeature1111);

Testcol1212=normTestFeatures1(:,12);
normTestcol1212=(Testcol1212-mean(colfeature1212))/std(colfeature1212);

Testcol1313=normTestFeatures1(:,13);
normTestcol1313=(Testcol1313-mean(colfeature1313))/std(colfeature1313);


Testnormalizedfeatures1=[normTestcol11,normTestcol22,normTestcol33,normTestcol44,normTestcol55,normTestcol66,normTestcol77,normTestcol88,normTestcol99,normTestcol1010,normTestcol1111,normTestcol1212,normTestcol1313];

DN1=dist2(normalizedfeatures1,normalizedfeatures1);
imagesc(DN1);
[DN1_sorted, DN1_index] = sort (DN1,2);
Predicted_Classes1=class(DN1_index(:,2:6));

Model=mode(Predicted_Classes1,2);
normtrainingaccuracymatimgk5=[class,Model];

normtrainingaccuracyimgk5=class==Model; 

TN1=dist2(Testnormalizedfeatures1,normalizedfeatures1);
[TN1_sorted,TN1_index]=sort(TN1,2);
normPredictedTesClass1=class(TN1_index(:,1:5));
Modelk5=mode(normPredictedTesClass1,2);

normtestaccuracymatimgk5=[Truetestclass,Modelk5];

normtestaccuracyimgk5=Truetestclass==Modelk5;

showcharlabels ('test.jpg', Modelk5, [1 2 3 4 5 6 7 8 9 10])

[ConfusionMat4,labels] = confusionmat(Truetestclass,Modelk5);