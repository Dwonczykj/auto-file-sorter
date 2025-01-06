import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { Box, TextField, CircularProgress, Typography, Paper } from '@mui/material';

interface RecipeVisualizerProps {
    query?: string;
    onQueryChange?: (query: string) => void;
}

export const RecipeVisualizer: React.FC<RecipeVisualizerProps> = ({
    query: initialQuery,
    onQueryChange
}) => {
    const [plotData, setPlotData] = useState<any>(null);
    const [staticImage, setStaticImage] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [localQuery, setLocalQuery] = useState(initialQuery || '');

    useEffect(() => {
        const fetchVisualization = async () => {
            try {
                setLoading(true);
                setError(null);

                const response = await fetch(`/api/recipes/visualize?query=${encodeURIComponent(localQuery)}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                setPlotData(data.interactive_plot);
                setStaticImage(data.static_image);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred');
                console.error('Error fetching visualization:', err);
            } finally {
                setLoading(false);
            }
        };

        const debounceTimer = setTimeout(() => {
            if (localQuery) {
                fetchVisualization();
            }
        }, 500); // Debounce API calls

        return () => clearTimeout(debounceTimer);
    }, [localQuery]);

    const handleQueryChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newQuery = event.target.value;
        setLocalQuery(newQuery);
        onQueryChange?.(newQuery);
    };

    return (
        <Box sx={{ p: 2 }}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                <TextField
                    fullWidth
                    label="Search Query"
                    variant="outlined"
                    value={localQuery}
                    onChange={handleQueryChange}
                    placeholder="Enter a recipe query to visualize similar recipes..."
                    sx={{ mb: 2 }}
                />

                {loading && (
                    <Box display="flex" justifyContent="center" my={2}>
                        <CircularProgress />
                    </Box>
                )}

                {error && (
                    <Typography color="error" sx={{ mb: 2 }}>
                        Error: {error}
                    </Typography>
                )}

                {plotData && !loading && (
                    <Box sx={{ mb: 2 }}>
                        <Plot
                            data={plotData.data}
                            layout={{
                                ...plotData.layout,
                                autosize: true,
                                margin: { l: 50, r: 50, t: 50, b: 50 },
                                hovermode: 'closest'
                            }}
                            style={{ width: '100%', height: '600px' }}
                            config={{
                                displayModeBar: true,
                                responsive: true,
                                scrollZoom: true
                            }}
                        />
                    </Box>
                )}

                {staticImage && !loading && (
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="h6" sx={{ mb: 1 }}>
                            Static Visualization
                        </Typography>
                        <img
                            src={staticImage}
                            alt="Recipe Embedding Visualization"
                            style={{
                                width: '100%',
                                maxHeight: '500px',
                                objectFit: 'contain'
                            }}
                        />
                    </Box>
                )}
            </Paper>
        </Box>
    );
}; 