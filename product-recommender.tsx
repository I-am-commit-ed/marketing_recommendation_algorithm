import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';

const ProductRecommender = () => {
  const [purchaseHistory, setPurchaseHistory] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Mock recommendation data - in production this would come from your Python backend
  const mockRecommendations = [
    "Classic White Shirt",
    "Black Jeans",
    "Leather Belt",
    "Canvas Sneakers",
    "Cotton T-Shirt"
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    // Simulate API call delay
    setTimeout(() => {
      setRecommendations(mockRecommendations);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Product Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Enter Purchase History (comma-separated product names)
              </label>
              <Input
                value={purchaseHistory}
                onChange={(e) => setPurchaseHistory(e.target.value)}
                placeholder="e.g., White Shirt, Blue Jeans, Black Shoes"
                className="w-full"
              />
            </div>
            <Button
              type="submit"
              disabled={isLoading}
              className="w-full"
            >
              {isLoading ? 'Generating Recommendations...' : 'Get Recommendations'}
            </Button>
          </form>

          {recommendations.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-2">Recommended Products</h3>
              <ScrollArea className="h-48 rounded-md border p-4">
                <ul className="space-y-2">
                  {recommendations.map((product, index) => (
                    <li key={index} className="p-2 bg-secondary rounded-md">
                      {product}
                    </li>
                  ))}
                </ul>
              </ScrollArea>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ProductRecommender;
