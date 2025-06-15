import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'goals_file': 'goal_scorers.csv',
    'results_file': 'match_results.csv',
    'output_dir': Path('output'),
    'top_scorers_limit': 10
}

@st.cache_data
def load_and_process_data(file_path: str, dataset_name: str) -> pd.DataFrame:
    """Load and preprocess CSV data with error handling."""
    try:
        if not Path(file_path).exists():
            st.error(f"File {file_path} not found!")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Process date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            initial_rows = len(df)
            df = df.dropna(subset=['date'])
            dropped_rows = initial_rows - len(df)
            
            if dropped_rows > 0:
                st.warning(f"Dropped {dropped_rows} rows with invalid dates from {dataset_name}")
            
            df['year'] = df['date'].dt.year
            df['decade'] = (df['year'] // 10) * 10
        
        return df
    
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        logger.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def create_matches_per_decade_chart(results: pd.DataFrame) -> pd.DataFrame:
    """Create and display matches per decade chart."""
    if results.empty:
        st.warning("No results data available for matches per decade analysis")
        return pd.DataFrame()
    
    matches_per_decade = (results.groupby('decade')
                         .size()
                         .reset_index(name='matches'))
    
    fig = px.bar(
        matches_per_decade, 
        x='decade', 
        y='matches',
        title='📅 Matches Per Decade',
        labels={'decade': 'Decade', 'matches': 'Number of Matches'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    return matches_per_decade

def create_unique_teams_chart(results: pd.DataFrame) -> pd.DataFrame:
    """Create and display unique teams per decade chart."""
    if results.empty or 'home_team' not in results.columns or 'away_team' not in results.columns:
        st.warning("No team data available for unique teams analysis")
        return pd.DataFrame()
    
    # More efficient approach using melt
    teams_data = pd.melt(
        results[['decade', 'home_team', 'away_team']], 
        id_vars=['decade'],
        value_vars=['home_team', 'away_team'],
        value_name='team'
    )
    
    unique_teams = (teams_data.groupby('decade')['team']
                   .nunique()
                   .reset_index(name='unique_teams'))
    
    fig = px.line(
        unique_teams, 
        x='decade', 
        y='unique_teams',
        title='🌍 Unique National Teams Per Decade',
        labels={'decade': 'Decade', 'unique_teams': 'Number of Unique Teams'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return unique_teams

def create_goals_per_match_chart(results: pd.DataFrame) -> pd.DataFrame:
    """Create and display average goals per match chart."""
    required_cols = ['home_score', 'away_score', 'year']
    if results.empty or not all(col in results.columns for col in required_cols):
        st.warning("No score data available for goals per match analysis")
        return pd.DataFrame()
    
    # Handle missing scores
    results_clean = results.dropna(subset=['home_score', 'away_score'])
    if len(results_clean) < len(results):
        st.info(f"Excluded {len(results) - len(results_clean)} matches with missing scores")
    
    results_clean['total_goals'] = results_clean['home_score'] + results_clean['away_score']
    goals_per_year = (results_clean.groupby('year')['total_goals']
                     .mean()
                     .reset_index())
    
    fig = px.line(
        goals_per_year, 
        x='year', 
        y='total_goals',
        title='⚽ Average Goals Per Match (Yearly)',
        labels={'year': 'Year', 'total_goals': 'Average Goals per Match'}
    )
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)
    
    return goals_per_year

def create_goal_scorers_analysis(goals: pd.DataFrame) -> dict:
    """Create comprehensive goal scorers analysis."""
    if goals.empty or 'scorer' not in goals.columns:
        st.warning("No scorer data available for goal scorers analysis")
        return {}
    
    # Remove null/empty scorers
    goals_clean = goals.dropna(subset=['scorer'])
    goals_clean = goals_clean[goals_clean['scorer'].str.strip() != '']
    
    if goals_clean.empty:
        st.warning("No valid scorer data found")
        return {}
    
    # Calculate scorer statistics
    scorer_stats = (goals_clean.groupby('scorer')
                   .agg({
                       'scorer': 'size',  # Total goals
                       'year': ['min', 'max'],  # Career span
                       'penalty': lambda x: x.sum() if 'penalty' in goals_clean.columns else 0,
                       'own_goal': lambda x: x.sum() if 'own_goal' in goals_clean.columns else 0
                   })
                   .round(2))
    
    # Flatten column names
    scorer_stats.columns = ['total_goals', 'first_year', 'last_year', 'penalties', 'own_goals']
    scorer_stats['career_span'] = scorer_stats['last_year'] - scorer_stats['first_year'] + 1
    scorer_stats = scorer_stats.sort_values('total_goals', ascending=False).reset_index()
    
    # Top scorers chart
    top_scorers = scorer_stats.head(CONFIG['top_scorers_limit'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            top_scorers, 
            x='total_goals', 
            y='scorer',
            orientation='h',
            title=f'🥇 Top {CONFIG["top_scorers_limit"]} Goal Scorers',
            labels={'total_goals': 'Number of Goals', 'scorer': 'Player'},
            color='total_goals',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top Scorer Stats")
        if not top_scorers.empty:
            top_player = top_scorers.iloc[0]
            st.metric("🏆 Leading Scorer", top_player['scorer'], f"{int(top_player['total_goals'])} goals")
            st.metric("📅 Career Span", f"{int(top_player['first_year'])}-{int(top_player['last_year'])}", 
                     f"{int(top_player['career_span'])} years")
            if 'penalty' in goals_clean.columns:
                st.metric("🎯 Penalties", int(top_player['penalties']))
    
    # Goals by decade for top scorers
    if len(top_scorers) > 0:
        st.subheader("🎯 Top Scorers by Decade")
        top_scorer_names = top_scorers['scorer'].tolist()
        top_scorer_goals = goals_clean[goals_clean['scorer'].isin(top_scorer_names)]
        
        decade_scorer_data = (top_scorer_goals.groupby(['decade', 'scorer'])
                             .size()
                             .reset_index(name='goals'))
        
        fig_decade = px.bar(
            decade_scorer_data,
            x='decade',
            y='goals',
            color='scorer',
            title='Goals by Top Scorers Across Decades',
            labels={'decade': 'Decade', 'goals': 'Number of Goals', 'scorer': 'Player'}
        )
        st.plotly_chart(fig_decade, use_container_width=True)
    
    return {
        'all_scorers': scorer_stats,
        'top_scorers': top_scorers
    }

def create_goal_scorers_list_display(goals: pd.DataFrame) -> pd.DataFrame:
    """Create a detailed goal scorers list for display and export."""
    if goals.empty or 'scorer' not in goals.columns:
        return pd.DataFrame()
    
    # Remove null/empty scorers
    goals_clean = goals.dropna(subset=['scorer'])
    goals_clean = goals_clean[goals_clean['scorer'].str.strip() != '']
    
    if goals_clean.empty:
        return pd.DataFrame()
    
    # Create comprehensive scorer statistics
    scorer_details = []
    
    for scorer in goals_clean['scorer'].unique():
        scorer_goals = goals_clean[goals_clean['scorer'] == scorer]
        
        stats = {
            'Rank': 0,  # Will be assigned later
            'Player Name': scorer,
            'Total Goals': len(scorer_goals),
            'First Goal Year': scorer_goals['year'].min(),
            'Last Goal Year': scorer_goals['year'].max(),
            'Career Span (Years)': scorer_goals['year'].max() - scorer_goals['year'].min() + 1,
            'Goals per Year': round(len(scorer_goals) / (scorer_goals['year'].max() - scorer_goals['year'].min() + 1), 2),
        }
        
        # Add penalty and own goal stats if available
        if 'penalty' in scorer_goals.columns:
            stats['Penalty Goals'] = scorer_goals['penalty'].sum()
            stats['Regular Goals'] = len(scorer_goals) - scorer_goals['penalty'].sum()
        
        if 'own_goal' in scorer_goals.columns:
            stats['Own Goals'] = scorer_goals['own_goal'].sum()
        
        # Add team information if available
        if 'team' in scorer_goals.columns:
            teams = scorer_goals['team'].unique()
            stats['Teams Played For'] = ', '.join(teams[:3])  # Show first 3 teams
            stats['Number of Teams'] = len(teams)
        
        scorer_details.append(stats)
    
    # Convert to DataFrame and sort by total goals
    scorers_df = pd.DataFrame(scorer_details)
    scorers_df = scorers_df.sort_values('Total Goals', ascending=False).reset_index(drop=True)
    scorers_df['Rank'] = range(1, len(scorers_df) + 1)
    
    # Reorder columns to put Rank first
    cols = ['Rank'] + [col for col in scorers_df.columns if col != 'Rank']
    scorers_df = scorers_df[cols]
    
    return scorers_df

def create_special_goals_charts(goals: pd.DataFrame) -> None:
    """Create and display own goals and penalty goals charts."""
    if goals.empty:
        st.warning("No goals data available for special goals analysis")
        return
    
    # Check for required columns
    has_own_goals = 'own_goal' in goals.columns
    has_penalties = 'penalty' in goals.columns
    
    if not (has_own_goals or has_penalties):
        st.warning("No own goal or penalty data available")
        return
    
    col1, col2 = st.columns(2)
    
    if has_own_goals:
        with col1:
            own_goals_data = goals[goals['own_goal'] == True]
            if not own_goals_data.empty:
                own_goals_yearly = (own_goals_data.groupby('year')
                                  .size()
                                  .reset_index(name='own_goals'))
                
                fig = px.line(
                    own_goals_yearly, 
                    x='year', 
                    y='own_goals',
                    title='🟥 Own Goals Per Year',
                    labels={'year': 'Year', 'own_goals': 'Number of Own Goals'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No own goals data found")
    
    if has_penalties:
        with col2:
            penalty_goals_data = goals[goals['penalty'] == True]
            if not penalty_goals_data.empty:
                penalty_goals_yearly = (penalty_goals_data.groupby('year')
                                      .size()
                                      .reset_index(name='penalty_goals'))
                
                fig = px.line(
                    penalty_goals_yearly, 
                    x='year', 
                    y='penalty_goals',
                    title='🎯 Penalty Goals Per Year',
                    labels={'year': 'Year', 'penalty_goals': 'Number of Penalty Goals'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No penalty goals data found")

def save_summary_data(data_dict: dict) -> None:
    """Save summary tables to CSV files."""
    try:
        CONFIG['output_dir'].mkdir(exist_ok=True)
        
        for filename, data in data_dict.items():
            if data is not None and not data.empty:
                filepath = CONFIG['output_dir'] / f"{filename}.csv"
                data.to_csv(filepath, index=False)
                logger.info(f"Saved {filename} to {filepath}")
        
        st.success(f"Summary data saved to {CONFIG['output_dir']} directory")
    
    except Exception as e:
        st.error(f"Error saving summary data: {str(e)}")
        logger.error(f"Error saving summary data: {str(e)}")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Football Analysis Dashboard",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("📊 150 Years of Football Analysis")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        goals = load_and_process_data(CONFIG['goals_file'], "Goals")
        results = load_and_process_data(CONFIG['results_file'], "Match Results")
    
    if goals.empty and results.empty:
        st.error("No data could be loaded. Please check your CSV files.")
        return
    
    # Display data info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Goals Recorded", len(goals) if not goals.empty else 0)
    with col2:
        st.metric("Total Matches Recorded", len(results) if not results.empty else 0)
    
    st.markdown("---")
    
    # Create visualizations
    summary_data = {}
    
    st.subheader("📈 Match Statistics")
    summary_data['matches_per_decade'] = create_matches_per_decade_chart(results)
    summary_data['teams_by_decade'] = create_unique_teams_chart(results)
    
    st.subheader("⚽ Goal Statistics")
    summary_data['goals_per_year'] = create_goals_per_match_chart(results)
    
    st.subheader("🏆 Goal Scorers Analysis")
    scorer_data = create_goal_scorers_analysis(goals)
    if scorer_data:
        summary_data.update(scorer_data)
    
    # Detailed Goal Scorers List
    st.subheader("📋 Complete Goal Scorers List")
    scorers_list = create_goal_scorers_list_display(goals)
    if not scorers_list.empty:
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_top_n = st.selectbox("Show top players:", [10, 25, 50, 100, "All"], index=0)
        with col2:
            min_goals = st.number_input("Minimum goals:", min_value=1, value=1, step=1)
        with col3:
            search_player = st.text_input("Search player name:")
        
        # Filter the list
        filtered_list = scorers_list[scorers_list['Total Goals'] >= min_goals].copy()
        
        if search_player:
            filtered_list = filtered_list[
                filtered_list['Player Name'].str.contains(search_player, case=False, na=False)
            ]
        
        if show_top_n != "All":
            filtered_list = filtered_list.head(show_top_n)
        
        # Display the table
        st.dataframe(
            filtered_list,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Player Name": st.column_config.TextColumn("Player Name", width="medium"),
                "Total Goals": st.column_config.NumberColumn("Total Goals", width="small"),
                "Goals per Year": st.column_config.NumberColumn("Goals/Year", width="small", format="%.2f"),
            }
        )
        
        # Add summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(scorers_list))
        with col2:
            st.metric("Players Shown", len(filtered_list))
        with col3:
            if not filtered_list.empty:
                st.metric("Average Goals", f"{filtered_list['Total Goals'].mean():.1f}")
        with col4:
            if not filtered_list.empty:
                st.metric("Total Goals Shown", filtered_list['Total Goals'].sum())
        
        summary_data['complete_scorers_list'] = scorers_list
    
    st.subheader("🎯 Special Goals Analysis")
    create_special_goals_charts(goals)
    
    # Save summary data
    st.markdown("---")
    if st.button("💾 Save Summary Data"):
        save_summary_data(summary_data)

if __name__ == "__main__":
    main()