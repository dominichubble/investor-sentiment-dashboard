# API Integration Documentation

## Overview

The frontend now integrates with the FastAPI backend to fetch and display real sentiment analysis data.

## API Service

Location: `src/services/api.ts`

The API service provides methods to interact with all backend endpoints:

### Data Endpoints

- **`getStatistics()`** - Get overall statistics
  - Total predictions
  - Sentiment distribution
  - Top stocks
  - Recent activity
  - Date range

- **`getPredictions(params)`** - Get historical predictions with filtering
  - Supports pagination
  - Filter by source, sentiment, date range

- **`getStockSentiment(ticker, params)`** - Get sentiment for specific stock

### Stock Endpoints

- **`getTrendingStocks(params)`** - Get most mentioned stocks
- **`getStockSentimentAggregated(ticker, params)`** - Get aggregated sentiment
- **`compareStocks(tickers, params)`** - Compare multiple stocks
- **`getStockStatistics()`** - Get stock-specific statistics

### Sentiment Endpoints

- **`analyzeSentiment(text, options)`** - Analyze single text
- **`batchAnalyzeSentiment(texts, options)`** - Batch analysis

### Health Check

- **`healthCheck()`** - Check API health status

## Configuration

### Environment Variables

Create `.env` file in frontend root:

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_ENV=development
```

### Backend Requirements

The backend API must be running at the configured URL (default: `http://localhost:8000`).

To start the backend:
```bash
cd backend
uvicorn api.main:app --reload
```

## Homepage Integration

The Homepage component now:

1. **Fetches real data** from `/api/v1/data/statistics` on mount
2. **Displays statistics**:
   - Net sentiment score (calculated from distribution)
   - Sentiment breakdown with percentages
   - Total documents analyzed
   - Number of stocks tracked
   - Top mentioned stocks
   - Recent activity (24h, 7d, 30d)

3. **Handles loading states** with spinner and message
4. **Handles errors** with retry button
5. **Auto-refreshes** when filters change

## Data Flow

```
Homepage Component
  ↓
API Service (api.ts)
  ↓
Axios HTTP Client
  ↓
FastAPI Backend (localhost:8000)
  ↓
Data Processing
  ↓
Frontend Display
```

## Error Handling

The frontend includes comprehensive error handling:

- **Network errors** - Shows error message with retry button
- **API errors** - Displays specific error details
- **Loading states** - Shows spinner while fetching
- **Empty states** - Handles missing data gracefully

## Data Transformation

The frontend transforms API responses for display:

### Net Sentiment Score

Calculated from sentiment distribution:
```typescript
score = (positive_percentage - negative_percentage) / 100
// Range: -1.00 to +1.00
```

### Sentiment Description

- **Strongly Positive**: > +0.5
- **Moderately Positive**: +0.2 to +0.5
- **Neutral**: -0.2 to +0.2
- **Moderately Negative**: -0.5 to -0.2
- **Strongly Negative**: < -0.5

### Summary Generation

Dynamic summary text generated from:
- Sentiment distribution
- Net sentiment score
- Top mentioned stocks
- Recent activity metrics
- Total statistics

## Future Enhancements

### Planned Features

1. **Time-series data** for trend charts
2. **Stock comparison** view
3. **Detailed stock pages**
4. **Real-time updates** via WebSockets
5. **Advanced filtering**:
   - By source (Reddit, Twitter, News)
   - By date range
   - By sentiment type
6. **Export functionality**
7. **User preferences** storage
8. **Dark mode** toggle

### API Additions Needed

- Time-series sentiment data for charts
- Source-specific breakdowns
- Hourly/daily aggregations
- Sentiment over time for stocks

## Testing

### Manual Testing

1. Start backend: `cd backend && uvicorn api.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:3000`
4. Verify:
   - Data loads correctly
   - Charts display properly
   - Error handling works
   - Filters trigger refresh

### API Testing

Test backend endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Statistics
curl http://localhost:8000/api/v1/data/statistics

# Predictions
curl "http://localhost:8000/api/v1/data/predictions?page=1&page_size=10"
```

## Troubleshooting

### "Unable to Load Data"

1. Check backend is running: `curl http://localhost:8000/health`
2. Verify API URL in `.env`
3. Check CORS settings in backend
4. Review browser console for errors

### Empty Data

1. Ensure backend has analyzed data
2. Check data storage is populated
3. Verify API returns data: `curl http://localhost:8000/api/v1/data/statistics`

### Charts Not Displaying

1. Check Recharts is installed: `npm list recharts`
2. Verify data format matches chart requirements
3. Review browser console for React errors

## Dependencies

- **axios**: HTTP client for API requests
- **recharts**: Chart library for visualizations
- **react**: UI framework
- **typescript**: Type safety

All dependencies are already included in `package.json`.
