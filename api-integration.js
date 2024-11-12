const handleSubmit = async (e) => {
  e.preventDefault();
  setIsLoading(true);
  
  try {
    const response = await fetch('http://localhost:5000/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        purchase_history: purchaseHistory
      })
    });
    
    const data = await response.json();
    
    if (response.ok) {
      setRecommendations(data.recommendations);
    } else {
      console.error('Error:', data.error);
      // Handle error appropriately in the UI
    }
  } catch (error) {
    console.error('Error:', error);
    // Handle error appropriately in the UI
  } finally {
    setIsLoading(false);
  }
};
