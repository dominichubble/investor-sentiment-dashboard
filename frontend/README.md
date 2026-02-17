# Investor Sentiment Dashboard - Frontend

TypeScript + React frontend for the Investor Sentiment Dashboard, generated from Figma designs using the Figma MCP server.

## 🚀 Getting Started

### Installation

```bash
cd frontend
npm install
```

### Development

Run the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Type Checking

```bash
npm run type-check
```

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## 📁 Project Structure

```
frontend/
├── index.html                    # Entry HTML file
├── vite.config.ts               # Vite configuration
├── tsconfig.json                # TypeScript configuration
├── package.json                 # Dependencies and scripts
└── src/
    ├── main.tsx                 # React entry point
    ├── App.tsx                  # Main App component
    ├── App.css                  # App styles
    ├── index.css                # Global styles
    ├── types/
    │   └── index.ts            # TypeScript type definitions
    ├── components/
    │   ├── DropdownButton/     # Interactive dropdown component
    │   │   ├── DropdownButton.tsx
    │   │   ├── DropdownButton.css
    │   │   └── index.ts
    │   ├── Navbar/             # Navigation bar component
    │   │   ├── Navbar.tsx
    │   │   ├── Navbar.css
    │   │   └── index.ts
    │   ├── MetricCard/         # Metric display card
    │   │   ├── MetricCard.tsx
    │   │   ├── MetricCard.css
    │   │   └── index.ts
    │   └── SummaryCard/        # Summary section card
    │       ├── SummaryCard.tsx
    │       ├── SummaryCard.css
    │       └── index.ts
    └── pages/
        └── Homepage/           # Main dashboard page
            ├── Homepage.tsx
            ├── Homepage.css
            └── index.ts
```

## 🎨 Components

### Homepage (`pages/Homepage`)

The main dashboard page that orchestrates all components and manages state:
- **State Management**: Handles asset filtering, timeframe selection, and data fetching
- **Data Loading**: Displays loading state while fetching data
- **Event Handling**: Manages user interactions with filters and cards

### Navbar (`components/Navbar`)

Navigation bar with interactive dropdowns:
- **Asset Selector**: Filter by all assets, crypto, stocks, or ETFs
- **Timeframe Selector**: Choose data range (7, 14, 30, or 90 days)
- **Responsive**: Adapts to mobile screens

### DropdownButton (`components/DropdownButton`)

Fully interactive dropdown component:
- **Click-to-open**: Opens dropdown menu on click
- **Close on outside click**: Automatically closes when clicking elsewhere
- **Keyboard accessible**: Supports Enter and Space key navigation
- **Visual feedback**: Shows open/closed state with animated arrow

### MetricCard (`components/MetricCard`)

Interactive card displaying metrics:
- **Dynamic colors**: Positive (green), Negative (red), Neutral (gray)
- **Trend indicators**: Visual up/down/neutral arrows
- **Clickable**: Can trigger navigation to detailed views
- **Hover effects**: Visual feedback on interaction

### SummaryCard (`components/SummaryCard`)

Summary section with embedded metric card:
- **Rich text**: Supports multi-paragraph summaries
- **Embedded visualization**: Includes a metric card for key data
- **Responsive layout**: Stacks on mobile devices

## 🔄 Interactive Features

### 1. **Dropdown Filtering**
- Select different asset classes or timeframes
- Automatically triggers data refresh
- Visual indication of current selection

### 2. **Card Interactions**
- Click any metric card to view details
- Hover effects for better UX
- Keyboard navigation support

### 3. **Loading States**
- Loading overlay when fetching data
- Prevents interaction during data refresh
- Smooth animations

### 4. **Responsive Design**
- Desktop: Full 4-column grid layout
- Tablet: 2-column grid
- Mobile: Single column stack

## 🔌 API Integration

The Homepage component includes a `fetchDashboardData` function that's ready for API integration:

```typescript
const fetchDashboardData = async () => {
  const response = await fetch(
    `/api/sentiment?asset=${selectedAsset.value}&timeframe=${selectedTimeframe.value}`
  );
  const data = await response.json();
  setDashboardData(data);
};
```

Currently uses mock data. Replace with actual API endpoint.

## 📊 Type Definitions

All types are defined in `src/types/index.ts`:

- `SentimentData`: Sentiment score information
- `MetricCardData`: Card display data
- `AssetFilter`: Asset filter options
- `SentimentBreakdown`: Positive/neutral/negative percentages
- `DashboardData`: Complete dashboard data structure

## 🛠️ Technologies

- **TypeScript** - Type safety and better developer experience
- **React 18** - UI library with hooks
- **Vite** - Fast build tool and dev server
- **React Router** - Navigation (installed, ready to use)
- **Axios** - HTTP client (installed, ready to use)
- **Recharts** - Charts library (installed, ready to use)
- **date-fns** - Date utilities (installed, ready to use)

## 📝 Next Steps

1. **Connect Real API**:
   - Update `fetchDashboardData` in Homepage.tsx
   - Replace mock data with actual endpoints
   - Add error handling

2. **Add Routing**:
   - Create detail pages for each metric
   - Implement navigation in `handleCardClick`

3. **Enhance Visualizations**:
   - Use Recharts for interactive graphs
   - Replace placeholder images with real charts

4. **Add More Features**:
   - Search functionality
   - Export data options
   - Dark mode toggle
   - User preferences

## 🎯 Features

- ✅ Full TypeScript support with strict mode
- ✅ Interactive dropdown filters
- ✅ Clickable metric cards
- ✅ Loading states and animations
- ✅ Responsive design (desktop/tablet/mobile)
- ✅ Component-based architecture
- ✅ Type-safe props and state
- ✅ Keyboard accessibility
- ✅ Mock data for development
- ✅ Ready for API integration

## 🐛 Development Tips

1. **Type Checking**: Run `npm run type-check` before committing
2. **Hot Reload**: Vite provides instant hot module replacement
3. **Component Testing**: Each component is isolated and testable
4. **Mock Data**: Modify `mockDashboardData` in Homepage.tsx for development
