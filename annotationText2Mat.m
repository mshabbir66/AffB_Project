%% Import data from text file.
% Script for importing data from the following text file:
%
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

clear all

%% Initialize variables.
path = '.\Dataset\';
file = 'Ses04F_impro03_candogan2.txt';
delimiter = '\t';

%% Format string for each line of text:
%   column1: text (%s)
%	column2: text (%s)
%   column3: double (%f)
%	column4: double (%f)
%   column5: text (%s)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%f%f%s%[^\n\r]';

%% Open the text file.
fileID = fopen([path file],'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
type = dataArray{:, 1};
sT = dataArray{:, 3};
eT = dataArray{:, 4};
suffix = dataArray{:, 5};

%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;

Ses04.endTime=zeros(length(type));
for i=1:length(type)
    Ses04(i).type=type{i};
    Ses04(i).startTime=sT(i);
    Ses04(i).endTime=eT(i);
    Ses04(i).fileName=file(1:end-4);
end

save([path file(1:end-4)],'Ses04');