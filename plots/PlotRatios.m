clc,clear

% Create arrays
dataset = 0; % 0 - MNIST, 1 - KMNIST
sparsity_level = 0:10:90;
if dataset
    ratios = [0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000;
        0.0989, 0.9011, 0.0987, 0.9013, 0.0210, 0.9790;
        0.2043, 0.7957, 0.1608, 0.8392, 0.0410, 0.9590;
        0.3098, 0.6902, 0.2212, 0.7788, 0.0770, 0.9230;
        0.4150, 0.5850, 0.2842, 0.7158, 0.1020, 0.8980;
        0.5199, 0.4801, 0.3494, 0.6506, 0.1380, 0.8620;
        0.6240, 0.3760, 0.4203, 0.5797, 0.1730, 0.8270;
        0.7268, 0.2732, 0.5017, 0.4983, 0.2170, 0.7830;
        0.8271, 0.1729, 0.6021, 0.3979, 0.2810, 0.7190;
        0.9223, 0.0777, 0.7404, 0.2596, 0.3970, 0.6030];
else
    ratios = [0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000;
        0.1034, 0.8966, 0.0643, 0.9357, 0.0130, 0.9870;
        0.2082, 0.7918, 0.1302, 0.8698, 0.0300, 0.9700;
        0.3130, 0.6870, 0.1972, 0.8028, 0.0510, 0.9490;
        0.4176, 0.5824, 0.2651, 0.7349, 0.0710, 0.9290;
        0.5218, 0.4782, 0.3352, 0.6648, 0.1050, 0.8950;
        0.6251, 0.3749, 0.4130, 0.5870, 0.1420, 0.8580;
        0.7274, 0.2726, 0.4984, 0.5016, 0.1800, 0.8200;
        0.8281, 0.1719, 0.5956, 0.4044, 0.2330, 0.7670;
        0.9251, 0.0749, 0.7205, 0.2795, 0.3370, 0.6630];
end   

% Create a bar graph
pbar = bar(sparsity_level,ratios,'stacked','FaceColor','flat');
% Set bar colors
pbar(1).CData = [254,226,215]/255;
pbar(2).CData = [246,83,20]/255;
pbar(3).CData = [189,233,255]/255;
pbar(4).CData = [0,161,241]/255;
pbar(5).CData = [216,255,138]/255;
pbar(6).CData = [124,187,0]/255;

% Set the legend
H = legend('Pruned /',...
    'Preserved parameters (1st layer)',...
    'Pruned /',...
    'Preserved parameters (2nd layer)',...
    'Pruned /',...
    'Preserved parameters (3rd layer)');
H.NumColumns = 2;
set(H,'Interpreter','latex','FontSize',12,'location','northoutside','Orientation','horizontal');

% Label the x-axis and y-axis
Tx = xlabel('The sparsity level, $$\bar{\kappa}$$ (\%)','FontSize',14);
set(Tx,'Interpreter','latex');
Ty = ylabel('The ratio','FontSize',14);
set(Ty,'Interpreter','latex');

% Specify y-axis tick labels
yticks([0 0.5 1 1.5 2 2.5 3 3.5 4])
yticklabels({'0','0.5','1 (0)','0.5','1 (0)','0.5','1 (0)', '0.5', '1'})