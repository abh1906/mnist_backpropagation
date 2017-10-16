data=csvread('mnist_train.csv');
Y=data(:,1);
X=data(:,2:785);

theta1=rand(25,785)*2*0.12-0.12;
X=[X ones(60000,1)];
size(X);
X=X';


a1=theta1*X;
a1=1./(1+exp(-a1));
theta2=rand(10 ,26)*2*0.12-0.12;
a1=[ones(1,60000);a1];
a2=theta2*a1;

a2=1./(1+exp(-a2));

[s,a]=max(a2,[],2);


output=zeros(10,60000);

for i=1:60000
 
 output(Y(i)+1,i)=1;

 
end
for i=1:100
delta3=a2-output;
delta2=theta2'*delta3.*a1.*(1-a1);
tri2=zeros(10,26)+delta3*a1';
tri1=zeros(26,785)+delta2*X';
d2=tri2./60000;
d1=tri1./60000;
d1(1,:)=[];
theta1=theta1-0.6*d1;
theta2=theta2-0.6*d2;

a1=theta1*X;
a1=1./(1+exp(-a1));

a1=[ones(1,60000);a1];
a2=theta2*a1;

a2=1./(1+exp(-a2));
end

data2=csvread('mnist_test.csv');
X2=data2(:,2:785);
Y2=data2(:,1);
X2=[X2 ones(10000,1)];
X2=X2';
a1=theta1*X2;
a1=1./(1+exp(-a1));

a1=[ones(1,10000);a1];
a2=theta2*a1;

a2=1./(1+exp(-a2));


a2=a2';
[s,a]=max(a2,[],2);
a=a-1;


fprintf('\nTraining Set Accuracy: %f\n', mean(double(a==Y2)) * 100);

