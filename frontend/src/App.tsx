import React, { useState, useEffect } from 'react';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { Tabs, Tab, Box, TextField, Slider, Typography, Paper, Container } from '@mui/material';
import { debounce } from 'lodash';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface Stats {
  recipe_pages: {
    total: number;
    avg_probability: number;
    recipe_pages_count: number;
    recipe_listings_count: number;
  };
  valid_recipes: {
    total: number;
    avg_calories: number;
    avg_cooking_time: number;
  };
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function RecipeSearchResults() {
  const [tabValue, setTabValue] = useState(0);
  const [recipePages, setRecipePages] = useState([]);
  const [recipes, setRecipes] = useState([]);
  const [urlAnalysis, setUrlAnalysis] = useState([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    minProb: 0,
    pathType: '',
    search: '',
    minCalories: '',
    maxCalories: '',
    domain: ''
  });

  // Column definitions for each table
  const recipePagesColumns: GridColDef[] = [
    { field: 'url', headerName: 'URL', flex: 1, minWidth: 200 },
    { field: 'probability', headerName: 'Probability', width: 130 },
    { field: 'pathType', headerName: 'Path Type', width: 130 },
    {
      field: 'lastChecked',
      headerName: 'Last Checked',
      width: 180,
      valueFormatter: (params) => params ? new Date(params?.value ?? new Date().toLocaleString()).toLocaleString() : new Date().toLocaleString()
    },
    { field: 'contentHash', headerName: 'Content Hash', width: 200 }
  ];

  const recipesColumns: GridColDef[] = [
    { field: 'title', headerName: 'Title', flex: 1, minWidth: 200 },
    { field: 'url', headerName: 'URL', flex: 1, minWidth: 200 },
    { field: 'calories', headerName: 'Calories', width: 100 },
    { field: 'cookingTime', headerName: 'Cooking Time (min)', width: 150 },
    {
      field: 'ingredients',
      headerName: 'Ingredients',
      flex: 1,
      minWidth: 200,
      valueFormatter: (params) => params ? params.value.join(', ') : ''
    },
    {
      field: 'lastUpdated',
      headerName: 'Last Updated',
      width: 180,
      valueFormatter: (params) => params ? new Date(params.value).toLocaleString() : new Date().toLocaleString()
    }
  ];

  const urlAnalysisColumns: GridColDef[] = [
    { field: 'domain', headerName: 'Domain', flex: 1, minWidth: 150 },
    { field: 'path', headerName: 'Path', flex: 2, minWidth: 200 },
    { field: 'probability', headerName: 'Probability', width: 130 },
    { field: 'pathType', headerName: 'Path Type', width: 130 },
    {
      field: 'lastChecked',
      headerName: 'Last Checked',
      width: 180,
      valueFormatter: (params) => params ? new Date(params.value).toLocaleString() : new Date().toLocaleString()
    }
  ];

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      let response;
      switch (tabValue) {
        case 0:
          const url = `/api/recipe-pages?${new URLSearchParams({
            min_prob: filters.minProb.toString(),
            path_type: filters.pathType,
            limit: '100'
          })}`;
          console.log('\n=== Frontend Request ===');
          console.log('URL:', url);
          console.log('Tab:', tabValue);
          console.log('Filters:', filters);

          response = await fetch(url);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          console.log('Response Data:', data);
          setRecipePages(data);
          break;

        case 1:
          response = await fetch(`/api/recipes?${new URLSearchParams({
            min_calories: filters.minCalories,
            max_calories: filters.maxCalories,
            search: filters.search,
            limit: '100'
          })}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          setRecipes(await response.json());
          break;

        case 2:
          response = await fetch(`/api/url-analysis?${new URLSearchParams({
            domain: filters.domain,
            min_prob: filters.minProb.toString(),
            path_type: filters.pathType,
            limit: '100'
          })}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          // Remove the stream reading debug code
          setUrlAnalysis(await response.json());
          break;
      }
    } catch (error) {
      console.error('Error fetching data:', error);
      // Optionally show error to user via UI
    } finally {
      setLoading(false);
    }
  };

  // Debounced filter updates
  const updateFilters = debounce((newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, 500);

  useEffect(() => {
    fetchData();
  }, [filters, tabValue]);

  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <Container maxWidth="xl" sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{
        width: '100%',
        p: 3,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
      }}>
        <Typography variant="h4" gutterBottom>
          Recipe Search Results
        </Typography>

        {stats && (
          <Paper sx={{ p: 2, mb: 2, flexShrink: 0 }}>
            <Typography variant="h6" gutterBottom>Statistics</Typography>
            <Box sx={{ display: 'flex', gap: 4 }}>
              <Box>
                <Typography variant="subtitle1">Recipe Pages</Typography>
                <Typography>Total: {stats.recipe_pages.total}</Typography>
                <Typography>Avg Probability: {stats.recipe_pages.avg_probability?.toFixed(2)}</Typography>
                <Typography>Recipe Pages: {stats.recipe_pages.recipe_pages_count}</Typography>
                <Typography>Recipe Listings: {stats.recipe_pages.recipe_listings_count}</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle1">Valid Recipes</Typography>
                <Typography>Total: {stats.valid_recipes.total}</Typography>
                <Typography>Avg Calories: {stats.valid_recipes.avg_calories?.toFixed(0)}</Typography>
                <Typography>Avg Cooking Time: {stats.valid_recipes.avg_cooking_time?.toFixed(0)} min</Typography>
              </Box>
            </Box>
          </Paper>
        )}

        <Tabs
          value={tabValue}
          onChange={(_, newValue) => setTabValue(newValue)}
          sx={{ flexShrink: 0 }}
        >
          <Tab label="Recipe Pages" />
          <Tab label="Valid Recipes" />
          <Tab label="URL Analysis" />
        </Tabs>

        <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
          <TabPanel value={tabValue} index={0}>
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              height: '100%'
            }}>
              <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center', flexShrink: 0 }}>
                <TextField
                  label="Path Type"
                  select
                  SelectProps={{ native: true }}
                  onChange={(e) => updateFilters({ pathType: e.target.value })}
                >
                  <option value="">All</option>
                  <option value="recipe_page">Recipe Page</option>
                  <option value="recipe_listing">Recipe Listing</option>
                </TextField>
                <Box sx={{ width: 200 }}>
                  <Typography gutterBottom>Min Probability</Typography>
                  <Slider
                    value={filters.minProb}
                    onChange={(_, value) => updateFilters({ minProb: value })}
                    valueLabelDisplay="auto"
                    step={0.1}
                    marks
                    min={0}
                    max={1}
                  />
                </Box>
              </Box>
              <Box sx={{
                flexGrow: 1,
                width: '100%',
                overflow: 'hidden',
                '& .MuiDataGrid-root': {
                  border: 'none',
                  '& .MuiDataGrid-cell': {
                    borderBottom: 1,
                    borderColor: 'divider',
                    whiteSpace: 'normal',
                    lineHeight: 'normal',
                    padding: 1
                  },
                  '& .MuiDataGrid-columnHeaders': {
                    borderBottom: 2,
                    borderColor: 'divider',
                    bgcolor: 'background.paper'
                  }
                }
              }}>
                <DataGrid
                  rows={recipePages}
                  columns={recipePagesColumns}
                  loading={loading}
                  pagination
                  getRowId={(row) => row.url}
                  initialState={{
                    pagination: { paginationModel: { pageSize: 25 } },
                  }}
                  pageSizeOptions={[25, 50, 100]}
                  disableColumnMenu
                  autoHeight={false}
                  density="comfortable"
                  getRowHeight={() => 'auto'}
                  sx={{
                    height: '100%',
                    width: '100%',
                    '& .MuiDataGrid-cell': {
                      maxHeight: 'none !important',
                    }
                  }}
                />
              </Box>
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
              <TextField
                label="Search"
                onChange={(e) => updateFilters({ search: e.target.value })}
              />
              <TextField
                label="Min Calories"
                type="number"
                onChange={(e) => updateFilters({ minCalories: e.target.value })}
              />
              <TextField
                label="Max Calories"
                type="number"
                onChange={(e) => updateFilters({ maxCalories: e.target.value })}
              />
            </Box>
            <Box sx={{ height: '600px', width: '100%' }}>
              <DataGrid
                rows={recipes}
                columns={recipesColumns}
                loading={loading}
                pagination
                getRowId={(row) => row.url}
                initialState={{
                  pagination: { paginationModel: { pageSize: 25 } },
                }}
                pageSizeOptions={[25, 50, 100]}
                disableColumnMenu
                autoHeight={false}
                sx={{
                  '& .MuiDataGrid-cell': {
                    whiteSpace: 'normal',
                    lineHeight: 'normal',
                    maxHeight: 'none !important',
                  },
                }}
              />
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
              <TextField
                label="Domain"
                onChange={(e) => updateFilters({ domain: e.target.value })}
              />
              <TextField
                label="Path Type"
                select
                SelectProps={{ native: true }}
                onChange={(e) => updateFilters({ pathType: e.target.value })}
              >
                <option value="">All</option>
                <option value="recipe_page">Recipe Page</option>
                <option value="recipe_listing">Recipe Listing</option>
              </TextField>
              <Box sx={{ width: 200 }}>
                <Typography gutterBottom>Min Probability</Typography>
                <Slider
                  value={filters.minProb}
                  onChange={(_, value) => updateFilters({ minProb: value })}
                  valueLabelDisplay="auto"
                  step={0.1}
                  marks
                  min={0}
                  max={1}
                />
              </Box>
            </Box>
            <Box sx={{ height: '600px', width: '100%' }}>
              <DataGrid
                rows={urlAnalysis}
                columns={urlAnalysisColumns}
                loading={loading}
                pagination
                getRowId={(row) => `${row.domain}${row.path}`}
                initialState={{
                  pagination: { paginationModel: { pageSize: 25 } },
                }}
                pageSizeOptions={[25, 50, 100]}
                disableColumnMenu
                autoHeight={false}
                sx={{
                  '& .MuiDataGrid-cell': {
                    whiteSpace: 'normal',
                    lineHeight: 'normal',
                    maxHeight: 'none !important',
                  },
                }}
              />
            </Box>
          </TabPanel>
        </Box>
      </Box>
    </Container>
  );
} 