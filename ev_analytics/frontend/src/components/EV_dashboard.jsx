import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Battery, Zap, DollarSign, Leaf } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const API_BASE_URL = '/api';  // Adjust based on your Django setup

const MetricCard = ({ icon: Icon, title, value, unit, description }) => (
  <Card className="relative overflow-hidden">
    <CardHeader className="space-y-1">
      <CardTitle className="text-lg font-bold flex items-center gap-2">
        <Icon className="w-5 h-5" />
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold mb-2">
        {value.toFixed(1)}{unit}
      </div>
      <p className="text-sm text-gray-500">{description}</p>
    </CardContent>
  </Card>
);

const EVDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState('7');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${API_BASE_URL}/charging-analytics/?days=${timeRange}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      
      const jsonData = await response.json();
      setData(jsonData);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="p-6">Loading...</div>;
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Error loading dashboard: {error}
        </AlertDescription>
      </Alert>
    );
  }

  if (!data) {
    return null;
  }

  const { summaryMetrics, chargingData, batteryData } = data;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">EV Charging Analytics Dashboard</h1>
        
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">Last 24 Hours</SelectItem>
            <SelectItem value="7">Last 7 Days</SelectItem>
            <SelectItem value="30">Last 30 Days</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(summaryMetrics).map(([key, metric]) => {
          const icons = {
            peakLoad: Zap,
            averageBatteryHealth: Battery,
            averageCost: DollarSign,
            renewableEnergy: Leaf
          };
          
          return (
            <MetricCard
              key={key}
              icon={icons[key]}
              title={key.replace(/([A-Z])/g, ' $1').trim()}
              value={metric.value}
              unit={metric.unit}
              description={metric.description}
            />
          );
        })}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="battery">Battery Health</TabsTrigger>
          <TabsTrigger value="efficiency">Efficiency</TabsTrigger>
          <TabsTrigger value="costs">Costs</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>Load Profile</CardTitle>
            </CardHeader>
            <CardContent className="h-80">
              <ResponsiveContainer>
                <LineChart data={chargingData.loadProfile}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
                  />
                  <YAxis label={{ value: 'Load (kW)', angle: -90 }} />
                  <Tooltip
                    labelFormatter={(ts) => new Date(ts).toLocaleString()}
                  />
                  <Line type="monotone" dataKey="load" stroke="#2563eb" name="Actual Load" />
                  <Line type="monotone" dataKey="prediction" stroke="#dc2626" name="Predicted" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Similar TabsContent components for other tabs... */}
      </Tabs>
    </div>
  );
};

export default EVDashboard;