x = linspace(0, 10, 100);

y = 2 * x + 3;

figure;
plot(x, y, 'b-', 'LineWidth', 2);
xlabel('X-axis');
ylabel('Y-axis');
title('Basic MATLAB Plot');
grid on;
legend('2x + 3', 'Location', 'Northwest');
