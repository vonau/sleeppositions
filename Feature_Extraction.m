%function [mean, sumdiff, deltaminmax, ampHR, ampRes, skew, kurt] = Feature_Extraction(data)
%% Feature Extraction
%The function extracts features from a given cleaned and filtered dataset
%% load data
%load("sensordata_cleaned_all_0311.mat");

%% parameters

fs= 1000;
windowsize= 10; %seconds
window= 10*fs;
overlap= 2; %seconds
mean= zeros(length(data(:,1)),8);
sumdiff= zeros(length(data(:,1)),8);
deltaminmax= zeros(length(data(:,1)),8);
ampHR= zeros(length(data(:,1)),8);
ampRes= zeros(length(data(:,1)),8);

%% Loop to select single channels

for i= 1:8
    channel= data(:,i);
    %% 8x mean pressure

    mean_pressure(i)= mean(channel);
    

    %% 8x sum of the difference between adjecent elements

    sumdiff(i)= sum(diff(channel));

    %% 8x Delta min/max

    deltaminmax(i)=

    %% 8x Energy HR

    ampHR(i)=

    %% 8x Energy respiration

    ampRes(i)=

end

%% skewness of means 
skew= skewness(mean_pressure);

%% kurtosis of means
kurt= kurtosis(mean_pressure);

%% CoP
CoP= (-4)*mean(1) + (-3)*mean(2) + (-2)*mean(3) + (-1)*mean(4) + mean(5) + 2*mean(6) + 3*mean(7) + 4*mean(8);

%% Skewness of Energy HR
skew_HR= skewness(ampHR);

%% Kurtosis of Energy HR
kurt_HR= kurtosis(ampHR);

%% Center of activity HR
CoAH=

%% Skewness of Energy respiration
skew_Res= skewness(ampHR);

%% Kurtosis of Energy respiration
kurt_Res= kurtosis(ampHR);

%% Center of activity respiration
CoAR=

