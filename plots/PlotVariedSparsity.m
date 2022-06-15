clc,clear

% Create arrays
sparsity_level = 0:10:90;
test_error = [1.35	1.39	1.37	1.40	1.50	1.43	1.56	1.74	1.95	2.38
    1.39	1.36	1.38	1.45	1.65	1.46	1.47	1.67	1.80	2.34
    1.44	1.24	1.44	1.43	1.42	1.58	1.62	1.74	1.97	2.38
    1.31	1.43	1.42	1.35	1.48	1.52	1.47	1.51	1.90	2.39
    1.38	1.31	1.37	1.26	1.46	1.34	1.64	1.56	1.85	2.46
    1.28	1.39	1.34	1.25	1.43	1.38	1.41	1.51	1.92	2.24
    1.24	1.43	1.32	1.44	1.51	1.46	1.57	1.56	1.93	2.39
    1.38	1.40	1.36	1.43	1.45	1.43	1.56	1.81	1.91	2.41
    1.25	1.20	1.48	1.43	1.48	1.49	1.61	1.73	2.04	2.34
    1.31	1.38	1.42	1.32	1.39	1.42	1.64	1.57	1.89	2.36]; % Copy from results.xlsx
average = mean(test_error,1);

% Create a scatter plot
color = linspace(1,20,length(sparsity_level));
for ns = 1:10
    scatter(sparsity_level,test_error(ns,:),30,color,'filled'); hold on
end
% Create a 2-D line plot
plot(sparsity_level,average,'x-','Linewidth',1,'Markersize',10,'Color',[246,83,20]/255); grid on

% Set axis limits
axis([min(sparsity_level) max(sparsity_level) 1 2.5]);

% Specify x-axis tick values
set(gca,'xtick',min(sparsity_level):10:max(sparsity_level),'FontSize',12);
set(gca,'FontSize',12);

% Label the x-axis and y-axis
Tx = xlabel('The sparsity level, $$\bar{\kappa}$$ (\%)','FontSize',14);
set(Tx, 'Interpreter','latex');
Ty = ylabel('The test error (\%)','FontSize',14);
set(Ty, 'Interpreter','latex');

% Set the background
set(gcf,'unit','normalized','position',[0.2,0.2,0.35,0.29]);