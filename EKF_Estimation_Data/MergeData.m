files = dir('*.mat');
merged = struct();

for i = 1:length(files)
    fileName = files(i).name;
    baseName = erase(fileName, '.mat');  % 去掉扩展名作为字段名
    temp = load(fileName);  % 加载结构体
    merged.(baseName) = temp;  % 把整个结构体存进去
end

save('merged_all.mat', 'merged');
