
data=csvread('output.csv')
reshape(data, [40,80,3])

data=data(:,1:2)

t=linspace(0,2*pi);

figure
hold on;
lines=plot(data(:,1),data(:,2));
lines.Color(4)=0.1;
plot(220*cos(t), 220*sin(t));
fill(200*cos(t), 200*sin(t),[0.4, 0.4, 0.4]);
hold off
axis([-250,250,-250,250])
axis square
print(gcf,'visual.png','-dpng','-r1000');


