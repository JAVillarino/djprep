//! Rekordbox XML schema constants

/// XML version declaration
pub const XML_VERSION: &str = "1.0";

/// XML encoding
pub const XML_ENCODING: &str = "UTF-8";

/// DJ_PLAYLISTS version attribute
pub const PLAYLISTS_VERSION: &str = "1.0.0";

/// Product name for the XML
pub const PRODUCT_NAME: &str = "djprep";

/// Product version (from Cargo.toml)
pub const PRODUCT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Rekordbox track attribute names
pub mod attrs {
    pub const TRACK_ID: &str = "TrackID";
    pub const NAME: &str = "Name";
    pub const ARTIST: &str = "Artist";
    pub const ALBUM: &str = "Album";
    pub const GENRE: &str = "Genre";
    pub const LOCATION: &str = "Location";
    pub const TOTAL_TIME: &str = "TotalTime";
    pub const AVERAGE_BPM: &str = "AverageBpm";
    pub const TONALITY: &str = "Tonality";
    pub const DATE_ADDED: &str = "DateAdded";
    pub const BIT_RATE: &str = "BitRate";
    pub const SAMPLE_RATE: &str = "SampleRate";
    pub const COMMENTS: &str = "Comments";
    pub const RATING: &str = "Rating";
    pub const PLAY_COUNT: &str = "PlayCount";
}

/// Playlist node types
pub mod node_types {
    /// Root folder type
    pub const ROOT: &str = "0";
    /// Playlist type
    pub const PLAYLIST: &str = "1";
    /// Folder type
    pub const FOLDER: &str = "0";
}
