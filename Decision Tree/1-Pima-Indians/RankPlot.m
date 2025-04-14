clear variables;
clc;
close all;

ann_recalls = [68.55, 69.47, 19.24];
dt_recalls = [63.64, 63.45, 56.94];

all_recalls = [ann_recalls, dt_recalls];
model_labels = {'ANN-1', 'ANN-2', 'ANN-3', 'DT-1', 'DT-2', 'DT-3'};

[sorted_recalls, sort_idx] = sort(all_recalls, 'descend');
sorted_labels = model_labels(sort_idx);
sorted_types = contains(sorted_labels, 'ANN'); 

figure('Position', [100, 100, 800, 500]);

hold on;
ann_bar = bar(0, 0);
set(ann_bar, 'FaceColor', [0.3, 0.6, 0.9], 'Visible', 'off');
dt_bar = bar(0, 0);
set(dt_bar, 'FaceColor', [0.9, 0.5, 0.1], 'Visible', 'off');
hb = bar(1:length(sorted_recalls), sorted_recalls);
for i = 1:length(sorted_recalls)
    if sorted_types(i)  % If ANN model
        hb.FaceColor = 'flat';
        hb.CData(i,:) = [0.3, 0.6, 0.9]; 
    else  
        hb.FaceColor = 'flat';
        hb.CData(i,:) = [0.9, 0.5, 0.1]; 
    end
end
hold off;

for i = 1:length(sorted_recalls)
    text(i, sorted_recalls(i) + 1.5, sprintf('%.2f%%', sorted_recalls(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold')
end

title('Ranking of Recall Values for Diabetes Prediction Models', 'FontSize', 14);
ylabel('Recall (%)', 'FontSize', 12);
xlabel('Model Iteration', 'FontSize', 12);

set(gca, 'XTick', 1:length(sorted_labels), 'XTickLabel', sorted_labels);
xtickangle(45);
ylim([0, max(all_recalls) * 1.15]);
grid on;

legend([ann_bar, dt_bar], {'ANN Models', 'DT Models'}, 'Location', 'northeast');

text(0.05, 0.05, 'Recall = Percentage of actual diabetes cases correctly identified', ...
    'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', [0.97, 0.97, 0.97]);