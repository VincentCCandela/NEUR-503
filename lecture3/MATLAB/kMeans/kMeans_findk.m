% Make the three distributions
X = [randn(100,2)+1.5*ones(100,2);...
     randn(100,2)-2*ones(100,2);...
     randn(100,2)+[-3*ones(100,1) 2*ones(100,1)];];

% Define a range of k values to try
k_values = 2:5;

for k = k_values
    % Perform k-means clustering
    opts = statset('Display','final');
    [idx,ctrs] = kmeans(X, k, 'Distance','sqeuclidean', 'Replicates',5, 'Options',opts);

    % Plot the clusters
    figure;
    clf;
    colors = 'rbgym'; % Color for each cluster
    hold on;
    for i = 1:k
        plot(X(idx==i,1), X(idx==i,2), [colors(i) '.'], 'MarkerSize', 12);
    end
    plot(ctrs(:,1), ctrs(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
    plot(ctrs(:,1), ctrs(:,2), 'ko', 'MarkerSize', 12, 'LineWidth', 2);

    % Create legend entries
    legendEntries = strcat('Cluster ', string(1:k));
    legendEntries{end+1} = 'Centroids';
    legend(legendEntries, 'Location', 'NW');

    % Perform silhouette analysis
    figure;
    clf;
    [silh, h] = silhouette(X, idx, 'sqeuclidean');
    xlabel('Silhouette Value');
    ylabel('Cluster');
    title(['Silhouette for k-means with k=', num2str(k)]);
end
